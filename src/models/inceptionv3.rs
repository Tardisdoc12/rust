//--------------------------------------------------------------------------------------------------
// Filename: inceptionv3.rs
// Author: Jean Anquetil
// Date: 2026-08-24
//--------------------------------------------------------------------------------------------------

use opencv::core::Mat;
use opencv::prelude::*;
use opencv::imgproc;
use ort::session::Session;
use ort::value::Value;

use crate::models::model_core::ModelPipeline;
use crate::functions_::setup_model_avec_cache::setup_model_avec_cache;

//--------------------------------------------------------------------------------------------------

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

fn preprocess(image: &Mat) -> anyhow::Result<Vec<f32>> {
    // 1. Resize à 299x299
    let mut resized = Mat::default();
    imgproc::resize(
        image, &mut resized,
        opencv::core::Size::new(299, 299),
        0.0, 0.0, imgproc::INTER_LINEAR,
    )?;

    // 2. BGR -> RGB
    let mut rgb = Mat::default();
    imgproc::cvt_color(
        &resized, &mut rgb,
        imgproc::COLOR_BGR2RGB, 0,
        opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    let data: &[u8] = rgb.data_bytes()?;
    let width = rgb.cols() as usize;
    let height = rgb.rows() as usize;

    // 3. Conversion en NCHW normalisé : [1, 3, 299, 299]
    // OpenCV stocke en HWC (interleaved), il faut réorganiser en CHW (planaire)
    let mut tensor = vec![0f32; 3 * height * width];

    for y in 0..height {
        for x in 0..width {
            let pixel_idx = (y * width + x) * 3;
            let r = data[pixel_idx] as f32 / 255.0;
            let g = data[pixel_idx + 1] as f32 / 255.0;
            let b = data[pixel_idx + 2] as f32 / 255.0;

            // normalisation par canal
            let r_norm = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
            let g_norm = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
            let b_norm = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];

            // écriture en layout CHW : canal 0 = tout R, canal 1 = tout G, canal 2 = tout B
            tensor[0 * height * width + y * width + x] = r_norm;
            tensor[1 * height * width + y * width + x] = g_norm;
            tensor[2 * height * width + y * width + x] = b_norm;
        }
    }

    Ok(tensor)
}

//--------------------------------------------------------------------------------------------------

fn infer(session: &mut Session, input: Vec<f32>) -> anyhow::Result<Vec<f32>> {
    let input_tensor = Value::from_array(([1, 3, 299, 299], input))?;

    let outputs = session.run(ort::inputs!["input" => input_tensor])?;

    let output_tensor = outputs["output"].try_extract_tensor::<f32>()?;
    let output_data: Vec<f32> = output_tensor.1.to_vec(); // .0 = shape, .1 = data

    Ok(output_data)
}

//--------------------------------------------------------------------------------------------------

#[derive(Debug)]
#[allow(dead_code)]
pub struct ClassificationResult {
    pub class_label: String,
    pub score: f32,
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    // stabilité numérique : soustrait le max avant exp (évite overflow)
    let max_logit = logits.iter().cloned().fold(f32::MIN, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&x| x / sum).collect()
}

fn postprocess(raw_output: &[f32], classes: &[String]) -> anyhow::Result<ClassificationResult> {
    let probabilities = softmax(raw_output);

    let (class_idx, &score) = probabilities
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .ok_or_else(|| anyhow::anyhow!("Vecteur de probabilités vide"))?;

    let class_label = classes
        .get(class_idx)
        .cloned()
        .unwrap_or_else(|| class_idx.to_string());

    Ok(ClassificationResult { class_label, score })
}

//--------------------------------------------------------------------------------------------------

pub struct InceptionV3 {
    session: Option<Session>,
    classes: Vec<String>,
}

impl InceptionV3 {
    pub fn new() -> Self {
        InceptionV3 { session: None, classes: Vec::new() }
    }
}

impl ModelPipeline for InceptionV3 {
    type Output = ClassificationResult;

    fn setup_model(&mut self, model_path: &str, device: &str) -> anyhow::Result<()> {
        let session = setup_model_avec_cache(model_path, device)?;

        let classes_str = {
            let metadata = session.metadata()?;
            metadata.custom("classes")
                .ok_or_else(|| anyhow::anyhow!("classes introuvables dans le modèle"))?
        };

        self.classes = serde_json::from_str(&classes_str)?;
        self.session = Some(session);

        Ok(())
    }

    fn preprocess(&self, image: &Mat) -> anyhow::Result<Vec<f32>> {
        preprocess(image)
    }

    fn infer(&mut self, input: &[f32]) -> anyhow::Result<Vec<f32>> {
        let session = self.session.as_mut()
            .ok_or_else(|| anyhow::anyhow!("Modèle non chargé, appelle setup_model d'abord"))?;
        infer(session, input.to_vec())
    }

    fn postprocess(&self, raw_output: &[f32]) -> anyhow::Result<Self::Output> {
        postprocess(raw_output, &self.classes)
    }

    fn unload(&mut self) {
        self.session = None;
    }
}

//--------------------------------------------------------------------------------------------------
// End of file
//--------------------------------------------------------------------------------------------------