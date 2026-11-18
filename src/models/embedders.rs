//--------------------------------------------------------------------------------------------------
// Filename: embedders.rs
// Author: Jean Anquetil
// Date: Wednesday 18/11/2026
//--------------------------------------------------------------------------------------------------

use opencv::{core::{Mat, Size, CV_32F}, imgproc, prelude::*};
use ort::session::Session;
use ort::value::Value;

use crate::models::model_core::ModelPipeline;
use crate::functions_::setup_model_avec_cache::setup_model_avec_cache;

//--------------------------------------------------------------------------------------------------

const SIZE: i32 = 224;
const DIM: usize = 1152;
const IMG_LEN: usize = 3 * 224 * 224;

//--------------------------------------------------------------------------------------------------


pub struct Embedder {
    session: Option<Session>,
}

//--------------------------------------------------------------------------------------------------

impl Embedder {
    pub fn new() -> Self {
        Self { session: None }
    }
}

impl ModelPipeline for Embedder {
    type Output = Vec<f32>;

    fn setup_model(
        &mut self,
        model_path: &str,
        device: &str,
    ) -> anyhow::Result<()> {

        let session = setup_model_avec_cache(model_path, device)?;

        self.session = Some(session);

        Ok(())
    }

    fn unload(&mut self) {
        self.session = None;
    }

    fn preprocess(&self, image: &Mat) -> anyhow::Result<Vec<f32>> {
        // 1) vers RGB (3 canaux)
        let mut rgb = Mat::default();
        match image.channels() {
            3 => imgproc::cvt_color(image, &mut rgb, imgproc::COLOR_BGR2RGB, 0)?,
            4 => { // BGRA : composite sur blanc comme load_rgb en Python (galerie)
                let mut bgr = Mat::default();
                let mut chans = opencv::core::Vector::<Mat>::new();
                opencv::core::split(image, &mut chans)?;
                let alpha = chans.get(3)?;
                imgproc::cvt_color(image, &mut bgr, imgproc::COLOR_BGRA2BGR, 0)?;
                let mut bgrf = Mat::default(); bgr.convert_to(&mut bgrf, CV_32F, 1.0, 0.0)?;
                let mut a = Mat::default(); alpha.convert_to(&mut a, CV_32F, 1.0 / 255.0, 0.0)?;
                let mut a3 = Mat::default();
                imgproc::cvt_color(&a, &mut a3, imgproc::COLOR_GRAY2BGR, 0)?;
                let mut one_minus = Mat::default();
                opencv::core::subtract(&opencv::core::Scalar::all(1.0), &a3, &mut one_minus, &opencv::core::no_array(), -1)?;
                let mut fg = Mat::default(); opencv::core::multiply(&bgrf, &a3, &mut fg, 1.0, -1)?;
                let mut bg = Mat::default(); one_minus.convert_to(&mut bg, -1, 255.0, 0.0)?;
                let mut sum = Mat::default(); opencv::core::add(&fg, &bg, &mut sum, &opencv::core::no_array(), -1)?;
                let mut u8m = Mat::default(); sum.convert_to(&mut u8m, opencv::core::CV_8U, 1.0, 0.0)?;
                imgproc::cvt_color(&u8m, &mut rgb, imgproc::COLOR_BGR2RGB, 0)?;
            }
            1 => imgproc::cvt_color(image, &mut rgb, imgproc::COLOR_GRAY2RGB, 0)?,
            n => anyhow::bail!("nombre de canaux non géré: {n}"),
        }
        // 2) "squash" : 224x224 directement, sans conserver le ratio ni recadrer (comme open_clip SigLIP)
        //    PIL bicubic est antialiasé à la réduction ; INTER_AREA est le plus proche (mesuré : erreur moyenne
        //    ~0.2-1/255 contre 1-3.6/255 pour INTER_CUBIC). INTER_CUBIC seulement si on agrandit.
        let interp = if rgb.cols() >= SIZE && rgb.rows() >= SIZE { imgproc::INTER_AREA } else { imgproc::INTER_CUBIC };
        let mut small = Mat::default();
        imgproc::resize(&rgb, &mut small, Size::new(SIZE, SIZE), 0.0, 0.0, interp)?;
        // 3) (x/255 - 0.5)/0.5 = x/127.5 - 1
        let mut f = Mat::default();
        small.convert_to(&mut f, CV_32F, 1.0 / 127.5, -1.0)?;
        // 4) HWC -> CHW
        let hwc = f.data_typed::<f32>()?; // contigu après resize/convert_to
        let (h, w) = (SIZE as usize, SIZE as usize);
        let mut chw = vec![0f32; 3 * h * w];
        for i in 0..h * w { for c in 0..3 { chw[c * h * w + i] = hwc[i * 3 + c]; } }
        Ok(chw)
    }

    fn infer(&mut self, input: &[f32]) -> anyhow::Result<Vec<f32>> {
        anyhow::ensure!(!input.is_empty() && input.len() % IMG_LEN == 0, "taille d'entrée invalide: {}", input.len());
        let n = input.len() / IMG_LEN;
        let sess = self.session.as_mut().ok_or_else(|| anyhow::anyhow!("modèle non chargé"))?;
        let tensor = Tensor::from_array(([n, 3usize, 224, 224], input.to_vec().into_boxed_slice()))?;
        let outputs = sess.run(ort::inputs!["pixel_values" => tensor])?;
        let (shape, data) = outputs["embedding"].try_extract_tensor::<f32>()?;
        anyhow::ensure!(shape[1] as usize == DIM, "dimension inattendue: {:?}", shape);
        Ok(data.to_vec())
    }

    /// Le modèle ONNX normalise déjà ; on re-normalise par sécurité (fp16 / autre export), ligne par ligne.
    fn postprocess(&self, raw_output: &[f32]) -> anyhow::Result<Vec<f32>> {
        let mut v = raw_output.to_vec();
        for row in v.chunks_mut(DIM) {
            let n = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            row.iter_mut().for_each(|x| *x /= n);
        }
        Ok(v)
    }
}

impl Embedder {
    /// Une image -> 1152 floats normalisés.
    pub fn embed(&mut self, image: &Mat) -> anyhow::Result<Vec<f32>> {
        let x = self.preprocess(image)?;
        let raw = self.infer(&x)?;
        self.postprocess(&raw)
    }

    /// Plusieurs crops en un passage GPU (utiliser des lots de 16-64 pour borner la VRAM).
    pub fn embed_batch(&mut self, images: &[Mat]) -> anyhow::Result<Vec<Vec<f32>>> {
        let mut flat = Vec::with_capacity(images.len() * IMG_LEN);
        for im in images { flat.extend(self.preprocess(im)?); }
        let raw = self.infer(&flat)?;
        Ok(self.postprocess(&raw)?.chunks(DIM).map(|c| c.to_vec()).collect())
    }
}

/// kNN : galerie = Vec<f32> aplati (m*1152, normalisé, ex. embeddings.npy), requête normalisée.
pub fn top1(gallery: &[f32], q: &[f32]) -> (usize, f32) {
    gallery.chunks(DIM).enumerate()
        .map(|(i, g)| (i, g.iter().zip(q).map(|(a, b)| a * b).sum::<f32>()))
        .fold((0, f32::MIN), |b, c| if c.1 > b.1 { c } else { b })
}