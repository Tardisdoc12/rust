//--------------------------------------------------------------------------------------------------
// Filename: yolo_26.rs
// Author: Jean Anquetil
// Date: 2026-08-25
//--------------------------------------------------------------------------------------------------

use std::sync::Mutex;

use ort::session::Session;
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

const IMAGE_SIZE: i32 = 320;
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

pub fn crop_mat(image: &Mat, bbox: (i32, i32, i32, i32)) -> anyhow::Result<Mat> {
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
            if !keep[i] { continue; }
            for j in 0..n {
                if i == j || !keep[j] { continue; }

                let area_i = Self::box_area(&detections[i].bbox);
                let area_j = Self::box_area(&detections[j].bbox);
                if area_i <= 0.0 || area_j <= 0.0 { continue; }

                let inter = Self::intersection_area(&detections[i].bbox, &detections[j].bbox);
                if inter <= 0.0 { continue; }

                let overlap_j = inter / area_j;
                if overlap_j > 0.7 && area_i > area_j {
                    keep[j] = false;   // même comportement que le Python
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
        // Export YOLOv7 --end2end (ONNX_ORT) : [batch_id, x1, y1, x2, y2, class_id, score]
        const VALUES_PER_DETECTION: usize = 7;

        if raw_output.len() % VALUES_PER_DETECTION != 0 {
            return Err(anyhow::anyhow!(
                "Sortie YOLOv7 invalide : {} valeurs, attendu un multiple de {}",
                raw_output.len(), VALUES_PER_DETECTION
            ));
        }

        let info = self.letterbox_info.lock().unwrap()
            .ok_or_else(|| anyhow::anyhow!("letterbox_info absent"))?;
        let source_image = self.source_image.lock().unwrap().take()
            .ok_or_else(|| anyhow::anyhow!("source_image absent"))?;

        // Comme en Python : on ne garde que la détection la plus confiante
        let best = raw_output
            .chunks_exact(VALUES_PER_DETECTION)
            .filter(|c| c[6] >= CONF_THRESHOLD)
            .max_by(|a, b| a[6].total_cmp(&b[6]));

        let Some(c) = best else { return Ok(Vec::new()) };

        let bbox = self.unletterbox_bbox(
            &BoundingBox { x1: c[1], y1: c[2], x2: c[3], y2: c[4] }, &info);
        let class_id = c[5] as usize;
        let class_label = self.classes.get(class_id).cloned()
            .unwrap_or_else(|| format!("class_{}", class_id));

        let categorie = match DetectionClass::from_str(&class_label) {
            Ok(cat) => cat,
            Err(_) => {
                eprintln!("[price_tag] classe inconnue de DetectionClass : {class_label:?}");
                return Ok(Vec::new());
            }
        };

        let img_shape = (info.original_height as usize, info.original_width as usize);
        let rust_bbox = BBox::new(bbox.x1, bbox.y1, bbox.x2, bbox.y2, img_shape);
        let cropped = crop_mat(&source_image,
            (bbox.x1 as i32, bbox.y1 as i32, bbox.x2 as i32, bbox.y2 as i32))?;
        let mut mask = Mask::new();
        mask.setup_mat(cropped);

        let mut det = BigDetection::new(categorie, rust_bbox, mask, String::new());
        det.set_prediction(c[6], class_label);
        Ok(vec![det])
    }

    fn unload(&mut self) {
        self.session = None;
    }
}