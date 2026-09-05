//--------------------------------------------------------------------------------------------------
// Filename: yolo_26.rs
// Author: Jean Anquetil
// Date: 2026-08-25
//--------------------------------------------------------------------------------------------------

use std::sync::Mutex;

use ort::session::Session;
use ort::ep::cuda::CUDA;
use opencv::core::Mat;
use opencv::prelude::*;
use opencv::imgproc;

use crate::models::model_core::ModelPipeline;

//--------------------------------------------------------------------------------------------------

const IMAGE_SIZE: i32 = 640;
const CONF_THRESHOLD: f32 = 0.25;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Detection {
    pub bbox: BoundingBox,
    pub score: f32,
    pub class_id: usize,
    pub class_label: String,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Detections {
    pub detections: Vec<Detection>,
}

#[derive(Debug, Clone, Copy)]
pub struct LetterboxInfo {
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
    pub original_width: f32,
    pub original_height: f32,
}

pub struct YOLO26 {
    session: Option<Session>,
    classes: Vec<String>,
    letterbox_info: Mutex<Option<LetterboxInfo>>,
}

//--------------------------------------------------------------------------------------------------

impl YOLO26 {
    pub fn new() -> Self {
        YOLO26 {
            session: None,
            classes: Vec::new(),
            letterbox_info: Mutex::new(None),
        }
    }

    fn letterbox(&self, image: &Mat, target_size: i32) -> anyhow::Result<(Mat, LetterboxInfo)> {
        let original_width = image.cols() as f32;
        let original_height = image.rows() as f32;

        // Ratio unique (le plus petit des deux) pour ne jamais dépasser target_size
        let scale = (target_size as f32 / original_width)
            .min(target_size as f32 / original_height);

        let new_width = (original_width * scale).round() as i32;
        let new_height = (original_height * scale).round() as i32;

        // 1. Resize en conservant le ratio
        let mut resized = Mat::default();
        imgproc::resize(
            image, &mut resized,
            opencv::core::Size::new(new_width, new_height),
            0.0, 0.0, imgproc::INTER_LINEAR,
        )?;

        // 2. Calcul du padding pour centrer l'image dans le carré cible
        let pad_x = ((target_size - new_width) as f32) / 2.0;
        let pad_y = ((target_size - new_height) as f32) / 2.0;

        let top = pad_y.round() as i32;
        let bottom = target_size - new_height - top;
        let left = pad_x.round() as i32;
        let right = target_size - new_width - left;

        // 3. Ajout du padding gris (114,114,114 = standard Ultralytics)
        let mut padded = Mat::default();
        opencv::core::copy_make_border(
            &resized, &mut padded,
            top, bottom, left, right,
            opencv::core::BORDER_CONSTANT,
            opencv::core::Scalar::new(114.0, 114.0, 114.0, 0.0),
        )?;

        let info = LetterboxInfo {
            scale,
            pad_x: left as f32,
            pad_y: top as f32,
            original_width,
            original_height,
        };

        Ok((padded, info))
    }

    /// Remappe une bbox exprimée dans l'espace 640x640 letterboxé
    /// vers l'espace de l'image originale.
    fn unletterbox_bbox(&self, bbox: &BoundingBox, info: &LetterboxInfo) -> BoundingBox {
        let x1 = (bbox.x1 - info.pad_x) / info.scale;
        let y1 = (bbox.y1 - info.pad_y) / info.scale;
        let x2 = (bbox.x2 - info.pad_x) / info.scale;
        let y2 = (bbox.y2 - info.pad_y) / info.scale;

        // clamp pour rester dans les limites de l'image d'origine
        BoundingBox {
            x1: x1.max(0.0).min(info.original_width),
            y1: y1.max(0.0).min(info.original_height),
            x2: x2.max(0.0).min(info.original_width),
            y2: y2.max(0.0).min(info.original_height),
        }
    }

    fn filter_detections(
        &self,
        detections: Vec<Detection>,
    ) -> Vec<Detection> {
        let n = detections.len();

        if n <= 1 {
            return detections;
        }

        let mut keep = vec![true; n];

        for i in 0..n {
            if !keep[i] {
                continue;
            }

            for j in 0..n {
                if i == j || !keep[j] {
                    continue;
                }

                let area_i = Self::box_area(&detections[i].bbox);
                let area_j = Self::box_area(&detections[j].bbox);

                if area_i <= 0.0 || area_j <= 0.0 {
                    continue;
                }

                let inter = Self::intersection_area(
                    &detections[i].bbox,
                    &detections[j].bbox,
                );

                if inter <= 0.0 {
                    continue;
                }

                let overlap_i = inter / area_i;
                let overlap_j = inter / area_j;

                // i est une grande boîte qui contient fortement j.
                // On garde la petite boîte j.
                if area_i > area_j && overlap_j > 0.7 {
                    keep[i] = false;
                    break;
                }

                // j est une grande boîte qui contient fortement i.
                // On garde la petite boîte i.
                if area_j > area_i && overlap_i > 0.7 {
                    keep[j] = false;
                    continue;
                }
            }
        }

        detections
            .into_iter()
            .enumerate()
            .filter_map(|(index, detection)| {
                if keep[index] {
                    Some(detection)
                } else {
                    None
                }
            })
            .collect()
    }

    fn box_area(bbox: &BoundingBox) -> f32 {
        let width = (bbox.x2 - bbox.x1).max(0.0);
        let height = (bbox.y2 - bbox.y1).max(0.0);

        width * height
    }

    fn intersection_area(
        a: &BoundingBox,
        b: &BoundingBox,
    ) -> f32 {
        let x1 = a.x1.max(b.x1);
        let y1 = a.y1.max(b.y1);
        let x2 = a.x2.min(b.x2);
        let y2 = a.y2.min(b.y2);

        let width = (x2 - x1).max(0.0);
        let height = (y2 - y1).max(0.0);

        width * height
    }
}

impl ModelPipeline for YOLO26 {
    type Output = Detections;

    fn setup_model(
        &mut self,
        model_path: &str,
        device: &str,
    ) -> anyhow::Result<()> {
        let mut builder = Session::builder()
            .map_err(|e| anyhow::anyhow!(
                "Erreur création du builder : {}",
                e
            ))?;

        if device.eq_ignore_ascii_case("cuda") {
            builder = builder
                .with_execution_providers([
                    CUDA::default().build()
                ])
                .map_err(|e| anyhow::anyhow!(
                    "Erreur configuration CUDA : {}",
                    e
                ))?;
        }

        let session = builder
            .commit_from_file(model_path)
            .map_err(|e| anyhow::anyhow!(
                "Erreur chargement modèle '{}': {}",
                model_path,
                e
            ))?;

        /*
        * Ultralytics exporte les noms de classes
        * dans les métadonnées du modèle.
        */
        let classes_str = {
            let metadata = session.metadata()?;

            let names = metadata.custom("names").filter(|s| !s.is_empty());
            let classes = metadata.custom("classes").filter(|s| !s.is_empty());

            names.or(classes)
                .ok_or_else(|| anyhow::anyhow!("Impossible de trouver les noms de classes dans les métadonnées ONNX"))?
        };

        let classes: Vec<String> = serde_json::from_str(&classes_str)?;

        self.classes = classes;
        self.session = Some(session);

        Ok(())
    }


    fn preprocess(&self, image: &Mat) -> anyhow::Result<Vec<f32>> {
        let (letterboxed, info) = self.letterbox(image, IMAGE_SIZE)?;
        *self.letterbox_info.lock().unwrap() = Some(info); // sauvegarde pour postprocess

        let mut rgb = Mat::default();
        imgproc::cvt_color(
            &letterboxed, &mut rgb,
            imgproc::COLOR_BGR2RGB, 0,
            opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

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

                tensor[0 * height * width + y * width + x] = r;
                tensor[1 * height * width + y * width + x] = g;
                tensor[2 * height * width + y * width + x] = b;
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
                    "Modèle non chargé, appelle setup_model d'abord"
                )
            })?;

        let input_tensor =
            ort::value::Value::from_array((
                [1, 3, IMAGE_SIZE, IMAGE_SIZE],
                input.to_vec(),
            ))?;

        let outputs = session.run(
            ort::inputs!["images" => input_tensor]
        )?;

        let output_tensor =
            outputs[0].try_extract_tensor::<f32>()?;

        Ok(output_tensor.1.to_vec())
    }

    fn postprocess(&self, raw_output: &[f32]) -> anyhow::Result<Self::Output> {
        if raw_output.len() % 6 != 0 {
            return Err(anyhow::anyhow!(
                "Sortie YOLO26 invalide : {} valeurs, attendu un multiple de 6",
                raw_output.len()
            ));
        }

        let info = self.letterbox_info.lock().unwrap()
            .ok_or_else(|| anyhow::anyhow!("letterbox_info absent, preprocess doit être appelé avant postprocess"))?;

        let mut detections = Vec::new();

        for chunk in raw_output.chunks_exact(6) {
            let score = chunk[4];
            if score < CONF_THRESHOLD {
                continue;
            }

            let raw_bbox = BoundingBox {
                x1: chunk[0],
                y1: chunk[1],
                x2: chunk[2],
                y2: chunk[3],
            };

            // remappage vers les coordonnées de l'image d'origine
            let bbox = self.unletterbox_bbox(&raw_bbox, &info);

            let class_id = chunk[5] as usize;
            let class_label = self.classes
                .get(class_id)
                .cloned()
                .unwrap_or_else(|| format!("class_{}", class_id));

            detections.push(Detection { bbox, score, class_id, class_label });
        }

        let detections = self.filter_detections(detections);
        Ok(Detections { detections })
    }
}