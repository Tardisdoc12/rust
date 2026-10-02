//--------------------------------------------------------------------------------------------------
// Filename: efficientnetb2.rs
// Author: Jean Anquetil
// Date: 2026-08-25
//--------------------------------------------------------------------------------------------------

use ort::session::Session;
use opencv::core::Mat;
use opencv::prelude::*;

use crate::models::model_core::ModelPipeline;
use crate::functions_::{setup_model_avec_cache::setup_model_avec_cache, letterbox::letterbox_rgb};

//--------------------------------------------------------------------------------------------------
const FILL_RGB: [u8; 3] = [0, 0, 0];
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ClassificationResult {
    pub class_label: String,
    pub score: f32,
    pub energy: f32,
}

pub struct EfficientNetB2 {
    session: Option<Session>,
    classes: Vec<String>,
    temperature: f32,
}

//--------------------------------------------------------------------------------------------------

impl EfficientNetB2 {
    pub fn new() -> Self {
        EfficientNetB2 {
            session: None,
            classes: Vec::new(),
            temperature: 1.0,
        }
    }

    fn energy_score(logits: &[f32], temperature: f32) -> f32 {
        let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();

        let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = scaled.iter().map(|&x| (x - max).exp()).sum();
        let logsumexp = max + sum_exp.ln();

        -temperature * logsumexp
    }
}

//--------------------------------------------------------------------------------------------------

impl ModelPipeline for EfficientNetB2 {
    type Output = ClassificationResult;

    fn setup_model(&mut self, model_path: &str, device: &str) -> anyhow::Result<()> {
        let session = setup_model_avec_cache(model_path, device)?;

        let (classes_str, temp_str) = {
            let metadata = session.metadata()?;
            let classes_str = metadata.custom("classes")
                .ok_or_else(|| anyhow::anyhow!("classes introuvables dans le modèle"))?;
            let temp_str = metadata.custom("temperature")
                .unwrap_or_else(|| "1.0".to_string()); // valeur par défaut si absente
            (classes_str, temp_str)
        };

        self.classes = serde_json::from_str(&classes_str)?;
        self.temperature = temp_str.parse::<f32>()
            .unwrap_or(1.0);

        self.session = Some(session);
        Ok(())
    }

    fn preprocess(&self, image: &Mat) -> anyhow::Result<Vec<f32>> {
        let rgb = letterbox_rgb(image, 288, FILL_RGB)?;

        let data: &[u8] = rgb.data_bytes()?;
        let width = rgb.cols() as usize;
        let height = rgb.rows() as usize;

        let mut tensor = vec![0f32; 3 * height * width];

        for y in 0..height {
            for x in 0..width {
                let pixel_idx = (y * width + x) * 3;
                let r = data[pixel_idx] as f32 / 255.0;
                let g = data[pixel_idx + 1] as f32 / 255.0;
                let b = data[pixel_idx + 2] as f32 / 255.0;

                let r_norm = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
                let g_norm = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
                let b_norm = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];

                tensor[0 * height * width + y * width + x] = r_norm;
                tensor[1 * height * width + y * width + x] = g_norm;
                tensor[2 * height * width + y * width + x] = b_norm;
            }
        }

        Ok(tensor)
    }

    fn infer(&mut self, input: &[f32]) -> anyhow::Result<Vec<f32>> {
        let session = self.session.as_mut()
            .ok_or_else(|| anyhow::anyhow!("Modèle non chargé, appelle setup_model d'abord"))?;

        let input_tensor = ort::value::Value::from_array(([1, 3, 288, 288], input.to_vec()))?;
        let outputs = session.run(ort::inputs!["input" => input_tensor])?;

        let output_tensor = outputs["output"].try_extract_tensor::<f32>()?;
        Ok(output_tensor.1.to_vec())
    }

    fn postprocess(&self, raw_output: &[f32]) -> anyhow::Result<Self::Output> {
        
        let scaled_logits: Vec<f32> = raw_output.iter()
            .map(|&x| x / self.temperature)
            .collect();

        let probabilities = softmax(&scaled_logits);

        let (class_idx, &score) = probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .ok_or_else(|| anyhow::anyhow!("Vecteur de probabilités vide"))?;

        let class_label = self.classes
            .get(class_idx)
            .cloned()
            .unwrap_or_else(|| class_idx.to_string());

        let energy = Self::energy_score(raw_output, self.temperature);

        Ok(ClassificationResult { class_label, score, energy })
    }

    fn unload(&mut self) {
        self.session = None;
    }
}

//--------------------------------------------------------------------------------------------------

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().cloned().fold(f32::MIN, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&x| x / sum).collect()
}

//--------------------------------------------------------------------------------------------------
// End of file
//--------------------------------------------------------------------------------------------------