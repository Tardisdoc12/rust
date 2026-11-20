use ort::session::Session;
use ort::value::Value;
use opencv::core::Mat;
use opencv::prelude::*;
use opencv::imgproc;

use crate::detections::mask::Mask;
use crate::functions_::setup_model_avec_cache::setup_model_avec_cache;

//--------------------------------------------------------------------------------------------------

const ENCODER_SIZE: i32 = 1024; // input_height/input_width du modèle
const BATCH: usize = 16; // boîtes par appel du décodeur (num_labels) : 16 = bon compromis vitesse / mémoire GPU
const MASK_SCALE_FACTOR: i32 = 4; // encoder_input_size / scale_factor = résolution native du masque

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

//--------------------------------------------------------------------------------------------------

pub struct Sam2Processor {
    encoder: Option<Session>,
    decoder: Option<Session>,

    image_embed: Option<Vec<f32>>,
    high_res_feat0: Option<Vec<f32>>,
    high_res_feat1: Option<Vec<f32>>,

    orig_width: i32,
    orig_height: i32,
}

//--------------------------------------------------------------------------------------------------

impl Sam2Processor {
    pub fn new() -> Self {
        Sam2Processor {
            encoder: None,
            decoder: None,
            image_embed: None,
            high_res_feat0: None,
            high_res_feat1: None,
            orig_width: 0,
            orig_height: 0,
        }
    }

    pub fn setup_model(&mut self, encoder_path: &str, decoder_path: &str, device: &str) -> anyhow::Result<()> {
        self.encoder = Some(setup_model_avec_cache(encoder_path, device)?);
        self.decoder = Some(setup_model_avec_cache(decoder_path, device)?);
        Ok(())
    }

    pub fn unload(&mut self) {
        self.encoder = None;
        self.decoder = None;
        self.image_embed = None;
        self.high_res_feat0 = None;
        self.high_res_feat1 = None;
    }

    /// Encode l'image une seule fois — équivalent de .encode() côté Python.
    /// IMPORTANT : simple resize (pas de letterbox), normalisation ImageNet.
    pub fn set_image(&mut self, image: &Mat) -> anyhow::Result<()> {
        self.orig_height = image.rows();
        self.orig_width = image.cols();

        // BGR -> RGB
        let mut rgb = Mat::default();
        imgproc::cvt_color(
            image, &mut rgb,
            imgproc::COLOR_BGR2RGB, 0,
        )?;

        // resize DIRECT vers 1024x1024, aspect ratio non conservé (conforme au Python)
        let mut resized = Mat::default();
        imgproc::resize(
            &rgb, &mut resized,
            opencv::core::Size::new(ENCODER_SIZE, ENCODER_SIZE),
            0.0, 0.0, imgproc::INTER_LINEAR,
        )?;

        let data: &[u8] = resized.data_bytes()?;
        let size = ENCODER_SIZE as usize;
        let mut tensor = vec![0f32; 3 * size * size];

        for y in 0..size {
            for x in 0..size {
                let idx = (y * size + x) * 3;
                let r = data[idx] as f32 / 255.0;
                let g = data[idx + 1] as f32 / 255.0;
                let b = data[idx + 2] as f32 / 255.0;

                let r_norm = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
                let g_norm = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
                let b_norm = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];

                tensor[0 * size * size + y * size + x] = r_norm;
                tensor[1 * size * size + y * size + x] = g_norm;
                tensor[2 * size * size + y * size + x] = b_norm;
            }
        }

        let encoder = self.encoder.as_mut()
            .ok_or_else(|| anyhow::anyhow!("Encoder non chargé"))?;

        let input = Value::from_array(([1, 3, ENCODER_SIZE, ENCODER_SIZE], tensor))?;
        let outputs = encoder.run(ort::inputs!["image" => input])?;

        self.image_embed = Some(outputs["image_embed"].try_extract_tensor::<f32>()?.1.to_vec());
        self.high_res_feat0 = Some(outputs["high_res_feats_0"].try_extract_tensor::<f32>()?.1.to_vec());
        self.high_res_feat1 = Some(outputs["high_res_feats_1"].try_extract_tensor::<f32>()?.1.to_vec());

        Ok(())
    }

    /// Une seule boîte (garde la signature d'avant) : passe par le chemin groupé.
    pub fn predict_box(&mut self, bbox: (f32, f32, f32, f32)) -> anyhow::Result<Mask> {
        self.predict_boxes(&[bbox])?.into_iter().next().ok_or_else(|| anyhow::anyhow!("aucun masque renvoyé"))
    }

    /// Plusieurs boîtes (mêmes coordonnées et convention que predict_box), décodées par lots de BATCH :
    /// l'encodeur d'image n'est pas relancé, le décodeur tourne une fois par lot au lieu d'une fois par boîte.
    /// REQUIERT un décodeur ONNX dont la 1re dimension de point_coords / point_labels / mask_input est dynamique (num_labels).
    pub fn predict_boxes(&mut self, boxes: &[(f32, f32, f32, f32)]) -> anyhow::Result<Vec<Mask>> {
        let mut out = Vec::with_capacity(boxes.len());
        for chunk in boxes.chunks(BATCH) {
            out.extend(self.predict_chunk(chunk)?);
        }
        Ok(out)
    }

    fn predict_chunk(&mut self, boxes: &[(f32, f32, f32, f32)]) -> anyhow::Result<Vec<Mask>> {
        let n = boxes.len();
        let (ow, oh) = (self.orig_width, self.orig_height);
        let enc = ENCODER_SIZE as f32;

        // une boîte = 2 points (labels 2 = haut-gauche, 3 = bas-droite), ratios x et y SÉPARÉS
        let mut point_coords = Vec::with_capacity(4 * n);
        let mut point_labels = Vec::with_capacity(2 * n);
        for &(x1, y1, x2, y2) in boxes {
            point_coords.extend([x1 / ow as f32 * enc, y1 / oh as f32 * enc, x2 / ow as f32 * enc, y2 / oh as f32 * enc]);
            point_labels.extend([2.0f32, 3.0f32]);
        }
        let mask_res = ENCODER_SIZE / MASK_SCALE_FACTOR; // 256
        let mask_input = vec![0f32; n * (mask_res * mask_res) as usize];   // (num_labels, 1, 256, 256) zéros
        let has_mask_input = vec![0f32; 1];                                  // forme fixe (1,)

        let embed = self.image_embed.as_ref().ok_or_else(|| anyhow::anyhow!("set_image doit être appelé avant predict_box(es)"))?;
        let feat0 = self.high_res_feat0.as_ref().unwrap();
        let feat1 = self.high_res_feat1.as_ref().unwrap();
        let decoder = self.decoder.as_mut().ok_or_else(|| anyhow::anyhow!("Decoder non chargé"))?;

        let outputs = decoder.run(ort::inputs![
            "image_embed" => Value::from_array(([1, 256, 64, 64], embed.clone()))?,
            "high_res_feats_0" => Value::from_array(([1, 32, 256, 256], feat0.clone()))?,
            "high_res_feats_1" => Value::from_array(([1, 64, 128, 128], feat1.clone()))?,
            "point_coords" => Value::from_array(([n as i64, 2, 2], point_coords))?,
            "point_labels" => Value::from_array(([n as i64, 2], point_labels))?,
            "mask_input" => Value::from_array(([n as i64, 1, mask_res as i64, mask_res as i64], mask_input))?,
            "has_mask_input" => Value::from_array(([1], has_mask_input))?,
        ])?;

        // masks : (n, num_masks, H, W) ; iou_predictions : (n, num_masks)
        let (mshape, mdata) = outputs["masks"].try_extract_tensor::<f32>()?;
        let mshape: Vec<i64> = mshape.to_vec();
        anyhow::ensure!(mshape.len() == 4 && mshape[0] as usize == n,
            "le décodeur n'a pas renvoyé {n} jeux de masques (forme {:?}) : son axe num_labels est probablement fixe à 1", mshape);
        let (nm, mh, mw) = (mshape[1] as usize, mshape[2] as usize, mshape[3] as usize);
        let (_, iou) = outputs["iou_predictions"].try_extract_tensor::<f32>()?;

        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            // meilleur masque parmi num_masks, comme masks[np.argmax(scores)]
            let best = (0..nm).max_by(|&a, &b| iou[i * nm + a].partial_cmp(&iou[i * nm + b]).unwrap()).unwrap_or(0);
            let start = (i * nm + best) * mh * mw;
            result.push(self.mask_to_full(&mdata[start..start + mh * mw], mh, mw)?);
        }
        Ok(result)
    }

    /// Masque natif (logits mh x mw) -> masque binaire à la taille de l'image originale.
    /// Même résultat que « resize bilinéaire de toute l'image puis seuil > 0 », mais le calcul ne se fait que dans la zone
    /// où un pixel peut être positif : l'interpolation bilinéaire est une moyenne pondérée, donc hors du rectangle englobant
    /// des logits > 0 (élargi d'un pixel natif) le résultat est <= 0. (Écart mesuré en Python : 1 à 3 pixels sur 12-24 millions,
    /// dû à l'arithmétique à virgule fixe de warpAffine.)
    fn mask_to_full(&self, low: &[f32], mh: usize, mw: usize) -> anyhow::Result<Mask> {
        use opencv::core::{Rect, Scalar, Size, BORDER_REPLICATE, CV_8UC1};
        let (ow, oh) = (self.orig_width, self.orig_height);
        let mut full = Mat::zeros(oh, ow, CV_8UC1)?.to_mat()?;

        // rectangle englobant des logits > 0
        let (mut xmin, mut xmax, mut ymin, mut ymax) = (usize::MAX, 0usize, usize::MAX, 0usize);
        for y in 0..mh { for x in 0..mw {
            if low[y * mw + x] > 0.0 { xmin = xmin.min(x); xmax = xmax.max(x); ymin = ymin.min(y); ymax = ymax.max(y); }
        }}
        if xmin == usize::MAX {
            return Ok(Mask { mat: Mat::default(), _mat_bin: full });      // aucun pixel positif : masque vide
        }
        let (sx, sy) = (mw as f64 / ow as f64, mh as f64 / oh as f64);   // pas dans l'espace natif par pixel de sortie
        let lo = |i: usize, s: f64, dim: i32| (((i as f64 - 1.0 + 0.5) / s - 0.5).floor() as i32).clamp(0, dim - 1);
        let hi = |i: usize, s: f64, dim: i32| (((i as f64 + 1.0 + 0.5) / s - 0.5).ceil() as i32).clamp(0, dim - 1);
        let (x0, x1) = (lo(xmin, sx, ow), hi(xmax, sx, ow));
        let (y0, y1) = (lo(ymin, sy, oh), hi(ymax, sy, oh));
        let (rw, rh) = (x1 - x0 + 1, y1 - y0 + 1);

        let low_mat = Mat::new_rows_cols_with_data(mh as i32, mw as i32, low)?.try_clone()?;
        // transformation destination(zone) -> source(natif), centres de pixels comme cv::resize
        let m = Mat::from_slice_2d(&[[sx, 0.0, sx * (x0 as f64 + 0.5) - 0.5], [0.0, sy, sy * (y0 as f64 + 0.5) - 0.5]])?;
        let mut roi_f = Mat::default();
        imgproc::warp_affine(&low_mat, &mut roi_f, &m, Size::new(rw, rh),
            imgproc::INTER_LINEAR | imgproc::WARP_INVERSE_MAP, BORDER_REPLICATE, Scalar::default())?;
        let mut roi_bin_f = Mat::default();
        imgproc::threshold(&roi_f, &mut roi_bin_f, 0.0, 255.0, imgproc::THRESH_BINARY)?;
        let mut roi_bin = Mat::default();
        roi_bin_f.convert_to(&mut roi_bin, CV_8UC1, 1.0, 0.0)?;

        let mut dst = Mat::roi_mut(&mut full, Rect::new(x0, y0, rw, rh))?;
        roi_bin.copy_to(&mut dst)?;
        drop(dst);
        Ok(Mask { mat: Mat::default(), _mat_bin: full })
    }
}