//--------------------------------------------------------------------------------------------------
// Filename: sam2.rs
//--------------------------------------------------------------------------------------------------

use ort::session::Session;
use ort::value::Value;
use ort::ep::cuda::CUDA;
use opencv::core::Mat;
use opencv::prelude::*;
use opencv::imgproc;

use crate::detections::mask::Mask;

//--------------------------------------------------------------------------------------------------

const ENCODER_SIZE: i32 = 1024; // input_height/input_width du modèle
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
        let build_session = |path: &str| -> anyhow::Result<Session> {
            let mut builder = Session::builder()
                .map_err(|e| anyhow::anyhow!("Erreur builder : {}", e))?;
            if device.eq_ignore_ascii_case("cuda") {
                builder = builder
                    .with_execution_providers([CUDA::default().build()])
                    .map_err(|e| anyhow::anyhow!("Erreur config CUDA : {}", e))?;
            }
            builder.commit_from_file(path)
                .map_err(|e| anyhow::anyhow!("Erreur chargement '{}': {}", path, e))
        };

        self.encoder = Some(build_session(encoder_path)?);
        self.decoder = Some(build_session(decoder_path)?);
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
            opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
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

    /// Prédit un masque pour une bbox (coordonnées dans l'espace de l'image ORIGINALE, en pixels).
    /// Convention SAM : une box = 2 points, labels 2 (top-left) et 3 (bottom-right).
    pub fn predict_box(&mut self, bbox: (f32, f32, f32, f32)) -> anyhow::Result<Mask> {
        let (x1, y1, x2, y2) = bbox;

        // normalisation des points : ratio x et y SÉPARÉS (pas un scale unique, cohérent
        // avec le resize non-aspect-preserving de l'encoder)
        let norm_x = |x: f32| x / self.orig_width as f32 * ENCODER_SIZE as f32;
        let norm_y = |y: f32| y / self.orig_height as f32 * ENCODER_SIZE as f32;

        let point_coords = vec![
            norm_x(x1), norm_y(y1), // top-left
            norm_x(x2), norm_y(y2), // bottom-right
        ];
        let point_labels = vec![2.0f32, 3.0f32];

        // mask_input : (num_labels=1, 1, 256, 256) zéros
        let mask_res = ENCODER_SIZE / MASK_SCALE_FACTOR; // 256
        let mask_input = vec![0f32; (1 * 1 * mask_res * mask_res) as usize];
        // has_mask_input : forme fixe (1,), PAS (num_labels,)
        let has_mask_input = vec![0f32; 1];

        let embed = self.image_embed.as_ref()
            .ok_or_else(|| anyhow::anyhow!("set_image doit être appelé avant predict_box"))?;
        let feat0 = self.high_res_feat0.as_ref().unwrap();
        let feat1 = self.high_res_feat1.as_ref().unwrap();

        let decoder = self.decoder.as_mut()
            .ok_or_else(|| anyhow::anyhow!("Decoder non chargé"))?;

        let embed_val = Value::from_array(([1, 256, 64, 64], embed.clone()))?;
        let feat0_val = Value::from_array(([1, 32, 256, 256], feat0.clone()))?;
        let feat1_val = Value::from_array(([1, 64, 128, 128], feat1.clone()))?;
        let points_val = Value::from_array(([1, 2, 2], point_coords))?;
        let labels_val = Value::from_array(([1, 2], point_labels))?;
        let mask_input_val = Value::from_array(([1, 1, mask_res, mask_res], mask_input))?;
        let has_mask_val = Value::from_array(([1], has_mask_input))?;

        let outputs = decoder.run(ort::inputs![
            "image_embed" => embed_val,
            "high_res_feats_0" => feat0_val,
            "high_res_feats_1" => feat1_val,
            "point_coords" => points_val,
            "point_labels" => labels_val,
            "mask_input" => mask_input_val,
            "has_mask_input" => has_mask_val,
        ])?;

        // masks: (num_labels=1, num_masks, H, W) — on prend le batch 0
        let masks_tensor = outputs["masks"].try_extract_tensor::<f32>()?;
        let masks_shape: Vec<i64> = masks_tensor.0.to_vec();
        let masks_data = masks_tensor.1;

        let mask_h = masks_shape[2] as usize;
        let mask_w = masks_shape[3] as usize;

        let iou_tensor = outputs["iou_predictions"].try_extract_tensor::<f32>()?;
        let scores: Vec<f32> = iou_tensor.1.to_vec();

        // sélection du meilleur masque, comme masks[np.argmax(scores)]
        let best_idx = scores.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let mask_start = best_idx * mask_h * mask_w;
        let best_mask_raw = &masks_data[mask_start..mask_start + mask_h * mask_w];

        // resize direct du masque natif (ex: 256x256) vers la taille de l'image originale
        let mask_mat = Mat::new_rows_cols_with_data(
            mask_h as i32, mask_w as i32, best_mask_raw,
        )?.try_clone()?; // clone car la data source (slice) doit survivre sinon

        let mut resized_mask = Mat::default();
        imgproc::resize(
            &mask_mat, &mut resized_mask,
            opencv::core::Size::new(self.orig_width, self.orig_height),
            0.0, 0.0, imgproc::INTER_LINEAR,
        )?;

        // Binarisation directe du Mat f32 (équivalent de `score > 0.0 -> 255 else 0`)
        let mut mat_bin_f32 = Mat::default();
        imgproc::threshold(
            &resized_mask,
            &mut mat_bin_f32,
            0.0,
            255.0,
            imgproc::THRESH_BINARY,
        )?;

        // Conversion CV_32F -> CV_8U
        let mut mat_bin = Mat::default();
        mat_bin_f32.convert_to(&mut mat_bin, opencv::core::CV_8UC1, 1.0, 0.0)?;

        Ok(Mask{
            mat: Mat::default(),
            _mat_bin: mat_bin.try_clone()?,
        })
    }
}