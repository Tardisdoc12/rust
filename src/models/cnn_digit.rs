//--------------------------------------------------------------------------------------------------
// Filename: cnn_digit
// Author: Jean Anquetil
// Date: 2026-08-26
//--------------------------------------------------------------------------------------------------

use opencv::core::Mat;
use opencv::prelude::*;
use opencv::imgproc;
use ort::session::Session;
use ort::ep::cuda::CUDA;

use crate::models::model_core::ModelPipeline;

//--------------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DigitResult {
    pub digit: usize,
    pub confidence: f32,
}

pub struct CNNDigit {
    session: Option<Session>,
    classes: Vec<String>,
}

//--------------------------------------------------------------------------------------------------

impl CNNDigit {
    pub fn new() -> Self {
        CNNDigit {
            session: None,
            classes: (0..10).map(|i| i.to_string()).collect(),
        }
    }
}

impl ModelPipeline for CNNDigit {
    type Output = DigitResult;

    fn unload(&mut self) {
        self.session = None;
    }

    fn setup_model(
        &mut self,
        model_path: &str,
        device: &str,
    ) -> anyhow::Result<()> {

        let mut builder = Session::builder()
            .map_err(|e| {
                anyhow::anyhow!(
                    "Erreur création du builder : {}",
                    e
                )
            })?;

        if device.eq_ignore_ascii_case("cuda") {
            builder = builder
                .with_execution_providers([
                    CUDA::default().build()
                ])
                .map_err(|e| {
                    anyhow::anyhow!(
                        "Erreur configuration CUDA : {}",
                        e
                    )
                })?;
        }

        let session = builder
            .commit_from_file(model_path)
            .map_err(|e| {
                anyhow::anyhow!(
                    "Erreur chargement CNN '{}': {}",
                    model_path,
                    e
                )
            })?;

        self.session = Some(session);

        Ok(())
    }

    fn preprocess(
        &self,
        image: &Mat,
    ) -> anyhow::Result<Vec<f32>> {

        let mut resized = Mat::default();

        imgproc::resize(
            image,
            &mut resized,
            opencv::core::Size::new(32, 32),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;

        let mut rgb = Mat::default();

        imgproc::cvt_color(
            &resized,
            &mut rgb,
            imgproc::COLOR_BGR2RGB,
            0,
            opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        let data = rgb.data_bytes()?;

        let width = rgb.cols() as usize;
        let height = rgb.rows() as usize;

        let plane_size = width * height;

        let mut tensor =
            vec![0.0f32; 3 * plane_size];

        for y in 0..height {
            for x in 0..width {

                let pixel =
                    (y * width + x) * 3;

                let r =
                    data[pixel] as f32 / 255.0;

                let g =
                    data[pixel + 1] as f32 / 255.0;

                let b =
                    data[pixel + 2] as f32 / 255.0;

                // Normalize :
                //
                // (x - 0.5) / 0.5
                //
                // = 2x - 1

                tensor[
                    y * width + x
                ] = (r - 0.5) / 0.5;

                tensor[
                    plane_size + y * width + x
                ] = (g - 0.5) / 0.5;

                tensor[
                    2 * plane_size + y * width + x
                ] = (b - 0.5) / 0.5;
            }
        }

        Ok(tensor)
    }

    fn infer(
        &mut self,
        input: &[f32],
    ) -> anyhow::Result<Vec<f32>> {

        let session = self
            .session
            .as_mut()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Modèle CNN non chargé"
                )
            })?;

        let input_tensor =
            ort::value::Value::from_array((
                [1, 3, 32, 32],
                input.to_vec(),
            ))?;

        let outputs = session.run(
            ort::inputs![
                "images" => input_tensor
            ]
        )?;

        let output =
            outputs[0]
                .try_extract_tensor::<f32>()?;

        println!(
            "CNN output shape = {:?}",
            output.0
        );

        Ok(output.1.to_vec())
    }

    fn postprocess(
        &self,
        raw_output: &[f32],
    ) -> anyhow::Result<Self::Output> {

        if raw_output.len() != 10 {
            return Err(anyhow::anyhow!(
                "Sortie CNN invalide : {} valeurs, attendu 10",
                raw_output.len()
            ));
        }

        // --------------------------------------------------
        // Softmax
        // --------------------------------------------------

        let max_logit = raw_output
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);

        let exp_values: Vec<f32> = raw_output
            .iter()
            .map(|x| (*x - max_logit).exp())
            .collect();

        let sum_exp: f32 =
            exp_values.iter().sum();

        let probabilities: Vec<f32> =
            exp_values
                .iter()
                .map(|x| x / sum_exp)
                .collect();

        // --------------------------------------------------
        // Argmax
        // --------------------------------------------------

        let (digit, confidence) =
            probabilities
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a.partial_cmp(b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(idx, &prob)| (idx, prob))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Impossible de déterminer le chiffre"
                    )
                })?;

        Ok(DigitResult {
            digit,
            confidence,
        })
    }
}