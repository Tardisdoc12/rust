//--------------------------------------------------------------------------------------------------
// Filename: bindings/pydetectionpipeline.rs
//--------------------------------------------------------------------------------------------------

use std::sync::Mutex;
use std::thread;

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use numpy::PyReadonlyArray3;
use opencv::core::{Mat, Rect};
use opencv::prelude::MatTraitConst;
use tracing_subscriber::EnvFilter;

use crate::processor::processor::Processor;
use crate::models::model_core::ModelPipeline;
use crate::models::{
    cnn_digit::CNNDigit,
    inceptionv3::InceptionV3,
    efficientnetb2::EfficientNetB2,
    yolo_26::YOLO26,
    sam2::Sam2Processor,
    yolo_v7::Yolov7PriceTag,
};
use crate::detections::{mask::Mask, detection_class::DetectionClass, detections::Detection};
use crate::functions_::functions_ocr::{
    Detection as OcrDetection, BoundingBox, PriceResult, group_to_price_str,
};
use crate::functions_::utils::{safe_float, clean_double_dot, clean_thousand_dot};
use crate::pipeline::workflow::compute_homography_from_reference;

//--------------------------------------------------------------------------------------------------

fn to_py_err(e: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Verrouille l'appel COMPLET à `.process()` (preprocess + infer +
/// postprocess), pas seulement l'inférence — voir la note en tête de fichier.
fn process_locked<M: ModelPipeline>(
    processor: &Mutex<Processor<M>>,
    image: &Mat,
) -> anyhow::Result<M::Output> {
    processor
        .lock()
        .map_err(|_| anyhow::anyhow!("Mutex empoisonné (un thread a paniqué en tenant le verrou)"))?
        .process(image)
}

fn is_digit_class(label: &str) -> bool {
    label.len() == 1 && label.chars().next().map_or(false, |c| c.is_ascii_digit())
}

fn crop_from_bbox(mat: &Mat, bbox: &BoundingBox) -> anyhow::Result<Mat> {
    let largeur = mat.cols();
    let hauteur = mat.rows();
    let x1 = (bbox.x1 as i32).max(0);
    let y1 = (bbox.y1 as i32).max(0);
    let x2 = (bbox.x2 as i32).min(largeur);
    let y2 = (bbox.y2 as i32).min(hauteur);
    let rect = Rect::new(x1, y1, (x2 - x1).max(1), (y2 - y1).max(1));
    let roi = Mat::roi(mat, rect)?;
    roi.try_clone().map_err(Into::into)
}

const REFERENCE_REAL_CM: f32 = 10.0;

//--------------------------------------------------------------------------------------------------

#[pyclass(name = "DetectionPipeline")]
pub struct PyDetectionPipeline {
    processor_yolo: Mutex<Processor<YOLO26>>,
    processor_yolo_price_tag: Mutex<Processor<YOLO26>>,
    processor_inception: Mutex<Processor<InceptionV3>>,
    processor_efficientnet: Mutex<Processor<EfficientNetB2>>,
    processor_yolo_ocr: Mutex<Processor<Yolov7PriceTag>>,
    processor_cnn_digit: Mutex<Processor<CNNDigit>>,
    sam2_processor: Mutex<Sam2Processor>,
}

#[pymethods]
impl PyDetectionPipeline {
    #[new]
    fn new(
        yolo_26_path: &str,
        yolo_v7_path: &str,
        sam2_path_encoder: &str,
        sam2_path_decoder: &str,
        cnn_digit_path: &str,
        inceptionv3_path: &str,
        efficientnetb2_path: &str,
        yolo_26_price_tag_path: &str,
        device: &str,
    ) -> PyResult<Self> {
        // Chargement des 8 modèles EN PARALLÈLE (fichiers et sessions ONNX
        // indépendants, aucune synchronisation nécessaire entre eux).
        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("ort=warn"));
        let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
        let result: anyhow::Result<_> = thread::scope(|scope| {
            let h_yolo = scope.spawn(|| Processor::new(YOLO26::new(), yolo_26_path, device));
            let h_yolo_pt = scope.spawn(|| Processor::new(YOLO26::new(), yolo_26_price_tag_path, device));
            let h_incep = scope.spawn(|| Processor::new(InceptionV3::new(), inceptionv3_path, device));
            let h_effnet = scope.spawn(|| Processor::new(EfficientNetB2::new(), efficientnetb2_path, device));
            let h_ocr = scope.spawn(|| Processor::new(Yolov7PriceTag::new(), yolo_v7_path, device));
            let h_digit = scope.spawn(|| Processor::new(CNNDigit::new(), cnn_digit_path, device));
            let h_sam2 = scope.spawn(|| -> anyhow::Result<Sam2Processor> {
                let mut sam2 = Sam2Processor::new();
                sam2.setup_model(sam2_path_encoder, sam2_path_decoder, device)?;
                Ok(sam2)
            });

            Ok((
                h_yolo.join().map_err(|_| anyhow::anyhow!("thread yolo a paniqué"))??,
                h_yolo_pt.join().map_err(|_| anyhow::anyhow!("thread yolo_price_tag a paniqué"))??,
                h_incep.join().map_err(|_| anyhow::anyhow!("thread inception a paniqué"))??,
                h_effnet.join().map_err(|_| anyhow::anyhow!("thread efficientnet a paniqué"))??,
                h_ocr.join().map_err(|_| anyhow::anyhow!("thread yolo_ocr a paniqué"))??,
                h_digit.join().map_err(|_| anyhow::anyhow!("thread cnn_digit a paniqué"))??,
                h_sam2.join().map_err(|_| anyhow::anyhow!("thread sam2 a paniqué"))??,
            ))
        });

        let (
            processor_yolo,
            processor_yolo_price_tag,
            processor_inception,
            processor_efficientnet,
            processor_yolo_ocr,
            processor_cnn_digit,
            sam2_processor,
        ) = result.map_err(to_py_err)?;

        Ok(Self {
            processor_yolo: Mutex::new(processor_yolo),
            processor_yolo_price_tag: Mutex::new(processor_yolo_price_tag),
            processor_inception: Mutex::new(processor_inception),
            processor_efficientnet: Mutex::new(processor_efficientnet),
            processor_yolo_ocr: Mutex::new(processor_yolo_ocr),
            processor_cnn_digit: Mutex::new(processor_cnn_digit),
            sam2_processor: Mutex::new(sam2_processor),
        })
    }

    fn unload(&self) {
        let _ = self.processor_yolo.lock().map(|mut p| p.unload());
        let _ = self.processor_yolo_price_tag.lock().map(|mut p| p.unload());
        let _ = self.processor_inception.lock().map(|mut p| p.unload());
        let _ = self.processor_efficientnet.lock().map(|mut p| p.unload());
        let _ = self.processor_yolo_ocr.lock().map(|mut p| p.unload());
        let _ = self.processor_cnn_digit.lock().map(|mut p| p.unload());
        let _ = self.sam2_processor.lock().map(|mut p| p.unload());
    }

    /// Traite une image. Les modèles déjà chargés sont réutilisés — appelez
    /// cette méthode autant de fois que nécessaire sur le même objet.
    fn process(&self, image: PyReadonlyArray3<'_, u8>) -> PyResult<Vec<Detection>> {
        let mut image_rust = Mask::new();
        image_rust.setup_from_python(image)?;
        let img_shape = (image_rust.mat.rows() as usize, image_rust.mat.cols() as usize);

        // --- Étape 1 : détection principale
        let mut vec_detections = process_locked(&self.processor_yolo, &image_rust.mat)
            .map_err(to_py_err)?;

        let mut homography: Option<Mat> = None;
        for detection in vec_detections.iter_mut() {
            if detection.categorie == DetectionClass::Reference {
                if let Ok(Some((h, _cm_per_pixel))) = self.classify_reference(detection) {
                    homography = Some(h);
                }
            }
        }

        let mut new_etiquettes: Vec<Detection> = Vec::new();
        for detection in vec_detections.iter_mut() {
            match detection.categorie {
                DetectionClass::Produit => {
                    self.classify_produit(detection, homography.as_ref());
                }
                DetectionClass::Etiquette => {
                    self.classify_etiquette(detection);
                }
                DetectionClass::Publicity => {
                    if let Ok(Some(etiquette)) = self.classify_publicity(detection, img_shape) {
                        new_etiquettes.push(etiquette);
                    }
                }
                _ => {}
            }
        }

        vec_detections.extend(new_etiquettes);
        Ok(vec_detections)
    }
}

// Méthodes internes (pas exposées à Python) : dispatch par catégorie.
impl PyDetectionPipeline {
    fn predict_mask(&self, bbox: (f32, f32, f32, f32)) -> anyhow::Result<Mask> {
        self.sam2_processor
            .lock()
            .map_err(|_| anyhow::anyhow!("Mutex SAM2 empoisonné"))?
            .predict_box(bbox)
    }

    fn classify_produit(&self, detection: &mut Detection, h: Option<&Mat>) {
        let Ok(result_inception) = process_locked(&self.processor_inception, &detection.mask.mat) else { return };
        let Ok(result_efficient) = process_locked(&self.processor_efficientnet, &detection.mask.mat) else { return };
        let Ok(mask) = self.predict_mask(detection.bbox.xyxyn()) else { return };

        detection.mask._mat_bin = mask._mat_bin;

        let label_inception = result_inception.class_label;
        let score_inception = result_inception.score;
        let label_efficient = result_efficient.class_label;
        let score_efficient = result_efficient.score;
        let energy_efficient = result_efficient.energy;

        if score_inception >= 0.996 {
            detection.label = label_inception;
            detection.score = score_inception;
        } else if score_inception >= 0.57 {
            detection.label = if score_efficient >= 0.23 { label_efficient } else { "OOD".to_string() };
            detection.score = score_efficient;
        } else {
            detection.label = "OOD".to_string();
            detection.score = 0.0;
            if energy_efficient < -5.0 && score_efficient > 0.5 {
                detection.label = label_efficient;
                detection.score = score_efficient;
            }
        }

        // Taille réelle : seulement si on a une homographie
        if let Some(h) = h {
            if let Ok((width_cm, height_cm)) = detection.get_real_size_from_homography(h) {
                detection.set_size(width_cm, height_cm);
            }
        }
    }

    fn classify_etiquette(&self, detection: &mut Detection) {
        let Ok(candidate_boxes) = process_locked(&self.processor_yolo_price_tag, &detection.mask.mat) else { return };

        let crops: Vec<Mat> = if candidate_boxes.is_empty() {
            match detection.mask.mat.try_clone() {
                Ok(m) => vec![m],
                Err(_) => return,
            }
        } else {
            candidate_boxes
                .iter()
                .filter_map(|b| {
                    let (x1, y1, x2, y2) = b.bbox.xyxy();
                    detection
                        .mask
                        .get_subpart_mat((x1 as i32, y1 as i32, x2 as i32, y2 as i32))
                        .ok()
                })
                .collect()
        };

        let mut best_price_str = String::new();
        let mut best_price_val = f32::MIN;

        for crop in &crops {
            let price_str = self.compute_price_on_crop(crop);
            let value = safe_float(&price_str);
            if value > best_price_val {
                best_price_val = value;
                best_price_str = price_str;
            }
        }

        detection.price = safe_float(&best_price_str);
    }

    /// Renvoie l'étiquette (prix) trouvée dans la publicité, si elle existe.
    fn classify_publicity(
        &self,
        detection: &Detection,
        img_shape: (usize, usize),
    ) -> anyhow::Result<Option<Detection>> {
        let Ok(new_results) = process_locked(&self.processor_yolo_price_tag, &detection.mask.mat) else {
            return Ok(None);
        };

        let best_detection = new_results
            .iter()
            .filter(|d| d.categorie == DetectionClass::Price)
            .max_by(|a, b| a.score.total_cmp(&b.score))
            .cloned();

        let Some(mut best) = best_detection else { return Ok(None) };

        // offset du crop publicity dans l'image complète
        let (px1, py1, _, _) = detection.bbox.xyxy();
        best.bbox = best.bbox.translate(px1, py1, img_shape);
        best.categorie = DetectionClass::Etiquette;

        self.classify_etiquette(&mut best);
        Ok(Some(best))
    }

    fn classify_reference(&self, detection: &mut Detection) -> anyhow::Result<Option<(Mat, f32)>> {
        let mask = self.predict_mask(detection.bbox.xyxyn())?;
        detection.mask._mat_bin = mask._mat_bin;
        compute_homography_from_reference(&detection.mask, REFERENCE_REAL_CM)
    }


    fn compute_price_on_crop(&self, crop: &Mat) -> String {
        let price_result: PriceResult = match process_locked(&self.processor_yolo_ocr, crop) {
            Ok(r) => r,
            Err(_) => return String::new(),
        };

        let corrected_group: Vec<OcrDetection> = price_result
            .detections
            .into_iter()
            .map(|mut d| {
                if is_digit_class(&d.class_label) {
                    if let Ok(sub_crop) = crop_from_bbox(crop, &d.bbox) {
                        if let Ok(result_cnn) = process_locked(&self.processor_cnn_digit, &sub_crop) {
                            if d.score < result_cnn.confidence {
                                d.class_label = result_cnn.digit.to_string();
                                d.score = result_cnn.confidence;
                            }
                        }
                    }
                }
                d
            })
            .collect();

        let price_str = group_to_price_str(corrected_group);
        let price_str = clean_double_dot(&price_str);
        clean_thousand_dot(&price_str)
    }
}

//--------------------------------------------------------------------------------------------------
// End of file
//--------------------------------------------------------------------------------------------------