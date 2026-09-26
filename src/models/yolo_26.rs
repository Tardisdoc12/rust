//--------------------------------------------------------------------------------------------------
// Filename: yolo_26.rs
// Author: Jean Anquetil
// Date: 2026-08-25
//--------------------------------------------------------------------------------------------------

use std::sync::Mutex;

use ort::session::Session;
use ort::ep::cuda::CUDA;
use ort::ep::tensorrt::TensorRT;
use opencv::core::Mat;
use opencv::prelude::*;
use opencv::imgproc;
use std::str::FromStr;
use opencv::core::Rect;

use crate::detections::detections::Detection as BigDetection;
use crate::detections::bbox::BBox;
use crate::detections::mask::Mask;
use crate::detections::detection_class::DetectionClass;
use crate::models::model_core::ModelPipeline;
use crate::functions_::setup_model_avec_cache::setup_model_avec_cache;


//--------------------------------------------------------------------------------------------------

const IMAGE_SIZE: i32 = 640;
const CONF_THRESHOLD: f32 = 0.25;

#[derive(Debug, Clone)]
struct BoundingBox {  // plus de `pub`
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

#[derive(Debug, Clone)]
struct RawDetection {  // renommé, plus de `pub`, purement interne
    bbox: BoundingBox,
    score: f32,
    class_id: usize,
    class_label: String,
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
    source_image: Mutex<Option<Mat>>,  // NOUVEAU
}

fn crop_mat(image: &Mat, bbox: (i32, i32, i32, i32)) -> anyhow::Result<Mat> {
    let (x1, y1, x2, y2) = bbox;
    let largeur = image.cols();
    let hauteur = image.rows();

    let x1 = x1.max(0);
    let y1 = y1.max(0);
    let x2 = x2.min(largeur);
    let y2 = y2.min(hauteur);

    let rect = Rect::new(x1, y1, (x2 - x1).max(1), (y2 - y1).max(1));
    let roi = Mat::roi(image, rect)?;
    Ok(roi.try_clone()?)
}

impl YOLO26 {
    pub fn new() -> Self {
        YOLO26 {
            session: None,
            classes: Vec::new(),
            letterbox_info: Mutex::new(None),
            source_image: Mutex::new(None),  // NOUVEAU
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
        detections: Vec<RawDetection>,
    ) -> Vec<RawDetection> {
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
    type Output =  Vec<BigDetection>;

    fn setup_model(
        &mut self,
        model_path: &str,
        device: &str,
    ) -> anyhow::Result<()> {
        let session = setup_model_avec_cache(model_path, device)?;

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
        *self.letterbox_info.lock().unwrap() = Some(info);
        *self.source_image.lock().unwrap() = Some(image.try_clone()?);

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

        // .take() plutôt que .clone() : Mat n'implémente pas Clone (seulement try_clone),
        // donc on retire la valeur de l'Option au lieu de la copier
        let source_image = self.source_image.lock().unwrap().take()
            .ok_or_else(|| anyhow::anyhow!("source_image absent, preprocess doit être appelé avant postprocess"))?;

        let mut raw_detections = Vec::new();

        for chunk in raw_output.chunks_exact(6) {
            let score = chunk[4];
            if score < CONF_THRESHOLD {
                continue;
            }

            let raw_bbox = BoundingBox { x1: chunk[0], y1: chunk[1], x2: chunk[2], y2: chunk[3] };
            let bbox = self.unletterbox_bbox(&raw_bbox, &info);

            let class_id = chunk[5] as usize;
            let class_label = self.classes.get(class_id).cloned()
                .unwrap_or_else(|| format!("class_{}", class_id));

            raw_detections.push(RawDetection { bbox, score, class_id, class_label });
        }

        let raw_detections = self.filter_detections(raw_detections);

        // --- Conversion vers le type métier utilisé dans le reste du pipeline ---
        let img_shape = (info.original_height as usize, info.original_width as usize);

        let detections: Vec<BigDetection> = raw_detections
            .into_iter()
            .filter_map(|raw| {
                // Si le label ne correspond à aucune DetectionClass connue, on ignore
                // silencieusement cette détection plutôt que de faire planter tout le batch.
                let categorie = DetectionClass::from_str(&raw.class_label).ok()?;

                let bbox = BBox::new(raw.bbox.x1, raw.bbox.y1, raw.bbox.x2, raw.bbox.y2, img_shape);

                let cropped = crop_mat(
                    &source_image,
                    (raw.bbox.x1 as i32, raw.bbox.y1 as i32, raw.bbox.x2 as i32, raw.bbox.y2 as i32),
                ).ok()?;

                let mut mask = Mask::new();
                mask.setup_mat(cropped);

                // base_name reste vide ici : cette info (Store_point/Shelf_number) est
                // calculée plus haut dans le pipeline Python, pas disponible à ce niveau.
                let mut detection = BigDetection::new(categorie, bbox, mask, String::new());
                detection.set_prediction(raw.score, raw.class_label);

                Some(detection)
            })
            .collect();

        Ok(detections)
    }

    fn unload(&mut self) {
        self.session = None;
    }
}