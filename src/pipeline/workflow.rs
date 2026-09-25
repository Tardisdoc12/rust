//--------------------------------------------------------------------------------------------------
// Filename: workflow.rs
// Author: Jean Anquetil
// Date: 2026-09-10
//--------------------------------------------------------------------------------------------------

use std::sync::Arc;

use pyo3::prelude::*;
use numpy::PyArray3;
use numpy::PyReadonlyArray3;
use opencv::core::{Mat, Rect, Point2f, Vector};
use opencv::prelude::MatTraitConst;

use crate::processor::processor::Processor;
use crate::models::{
    cnn_digit::CNNDigit,
    inceptionv3::InceptionV3,
    efficientnetb2::EfficientNetB2,
    yolo_26::YOLO26,
    sam2::Sam2Processor,
    yolo_v7::Yolov7PriceTag
};

use crate::pipeline::pipeline::{Pipeline, PipelineStep, StepFn};
use crate::pipeline::pipeline_data::PipelineData;
use crate::detections::{mask::Mask, detection_class::DetectionClass, detections::Detection};
use crate::functions_::functions_ocr::{
    Detection as OcrDetection, BoundingBox, PriceResult, group_to_price_str,
};
use crate::functions_::utils::{safe_float, clean_double_dot, clean_thousand_dot};
use crate::tools_class::connecteur::ConnecteurServer;

//--------------------------------------------------------------------------------------------------

#[pyfunction]
pub fn workflows(
    image: PyReadonlyArray3<'_, u8>,
    yolo_26_path: &str,
    yolo_v7_path: &str,
    sam2_path_encoder: &str,
    sam2_path_decoder: &str,
    cnn_digit_path: &str,
    inceptionv3_path: &str,
    efficientnetb2_path: &str,
    yolo_26_price_tag_path: &str,
    device: &str,
)-> PyResult<Vec<Detection>> {

    let mut image_rust = Mask::new();
    image_rust.setup_from_python(image);

    let vec: Vec<PipelineStep> = vec![
        PipelineStep::Single(
            get_detections_from_image(
                yolo_26_path,
                yolo_26_price_tag_path,
                device,
            )
        ),
        PipelineStep::Single(
            compare_results_inception_efficientnet(
                inceptionv3_path,
                efficientnetb2_path,
                device,
            )
        ),
        PipelineStep::Single(
            compare_ocr_cnn_price(
                cnn_digit_path,
                yolo_v7_path,
                yolo_26_price_tag_path,
                device,
            )
        ),
        PipelineStep::Single(
            get_sam2(
                sam2_path_encoder,
                sam2_path_decoder,
                &image_rust.mat.clone(),
                device,
            )
        ),
        PipelineStep::Single(
            get_real_size_of_objects()
        ),
        PipelineStep::Single(
            get_real_label_of_objects()
        ),
    ];

    let input_data = PipelineData::Image(Arc::new(image_rust));

    let mut pipeline = Pipeline::new(vec);
    let result = pipeline.run(input_data);

    match result {
        PipelineData::Detections(detections) => {
            Ok(Arc::try_unwrap(detections).unwrap_or_else(|arc| (*arc).clone()))
        }
        _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Le pipeline ne s'est pas terminé sur des détections",
        )),
    }
}

//--------------------------------------------------------------------------------------------------

fn get_detections_from_image(
    yolo_26_path: &str,
    yolo_26_price_tag_path: &str,
    device: &str,
) -> StepFn {
    let yolo_model = YOLO26::new();
    let mut processor_yolo = Processor::new(
        yolo_model,
        yolo_26_path,
        device,
    ).expect("Échec de la création du processeur");

    let yolo_price_tag_model = YOLO26::new();
    let mut processor_yolo_price_tag = Processor::new(
        yolo_price_tag_model,
        yolo_26_price_tag_path,
        device,
    ).expect("Échec de la création du processeur");
    
    Box::new(move |input: PipelineData| {
        let PipelineData::Image(img) = input else {
            panic!("expected an image at this stage")
        };

        let img_shape = (img.mat.rows() as usize, img.mat.cols() as usize);

        let mut vec_detections = processor_yolo.process(&img.mat)
            .expect("Échec de la détection des objets dans l'image");

        let mut nouvelles_detections = Vec::new();

        vec_detections.iter().for_each(|detection| {
            if detection.categorie == DetectionClass::Publicity {
                let new_results = processor_yolo_price_tag.process(&detection.mask.mat)
                    .expect("Échec de la détection des étiquettes dans les publicités");

                let mut best_score = 0.0;
                let mut best_detection = None;

                new_results.iter().for_each(|new_detection| {
                    if new_detection.categorie != DetectionClass::Etiquette {
                        return;
                    }
                    if new_detection.score > best_score {
                        best_score = new_detection.score;
                        best_detection = Some(new_detection.clone());
                    }
                });

                if let Some(mut best) = best_detection {
                    // offset du crop publicity dans l'image complète
                    let (px1, py1, _, _) = detection.bbox.xyxy();

                    best.bbox = best.bbox.translate(px1, py1, img_shape);
                    best.categorie = DetectionClass::OtherEtiquette;

                    nouvelles_detections.push(best);
                }
            }
        });

        vec_detections.extend(nouvelles_detections);

        processor_yolo.unload();
        processor_yolo_price_tag.unload();
        return PipelineData::Detections(Arc::new(vec_detections))
    })
}

//--------------------------------------------------------------------------------------------------

fn compare_results_inception_efficientnet(
    inceptionv3_path: &str,
    efficientnetb2_path: &str,
    device: &str,
) -> StepFn {
    let inception_model = InceptionV3::new();
    let mut processor_inception = Processor::new(
        inception_model,
        inceptionv3_path,
        device,
    ).expect("Échec de la création du processeur");

    let efficientnet_model = EfficientNetB2::new();
    let mut processor_efficientnet = Processor::new(
        efficientnet_model,
        efficientnetb2_path,
        device,
    ).expect("Échec de la création du processeur");

    Box::new(move |input: PipelineData| {
        let PipelineData::Detections(detections) = input else {
            panic!("expected detections at this stage")
        };

        // On récupère un Vec mutable. try_unwrap évite une clone si on est
        // l'unique propriétaire de l'Arc (ce qui devrait être le cas ici).
        let mut vec_detections = Arc::try_unwrap(detections)
            .unwrap_or_else(|arc| (*arc).clone());

        vec_detections.iter_mut().for_each(|detection| {
            if detection.categorie != DetectionClass::Produit {
                return;
            }

            let result_inception =
                processor_inception.process(&detection.mask.mat)
                .expect("Échec de la classification InceptionV3");
            let result_efficient = 
                processor_efficientnet.process(&detection.mask.mat)
                .expect("Échec de la classification EfficientNetB2");

            let label_inception = result_inception.class_label;
            let score_inception = result_inception.score;

            let label_efficient = result_efficient.class_label;
            let score_efficient = result_efficient.score;

            if score_inception >= 0.996 {
                detection.label = label_inception;
                detection.score = score_inception;
            } else if score_inception >= 0.57 {
                detection.label = if score_efficient >= 0.23 {
                    label_efficient
                } else {
                    "OOD".to_string()
                };
                detection.score = score_efficient;
            } else {
                detection.label = "OOD".to_string();
                detection.score = 0.0;
            }
        });

        processor_inception.unload();
        processor_efficientnet.unload();

        PipelineData::Detections(Arc::new(vec_detections))
    })
}

//--------------------------------------------------------------------------------------------------

fn is_digit_class(label: &str) -> bool {
    label.len() == 1 && label.chars().next().map_or(false, |c| c.is_ascii_digit())
}

//--------------------------------------------------------------------------------------------------

/// Crop direct sur un Mat brut à partir d'une bbox pixel absolue (équivalent
/// de Mask::get_subpart_mat mais sans passer par un objet Mask).
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

//--------------------------------------------------------------------------------------------------

/// Équivalent de treat_box_to_price + clean_double_dot + clean_thousand_dot,
/// appliqué sur un crop déjà découpé.
fn compute_price_on_crop(
    crop: &Mat,
    processor_yolo_ocr: &mut Processor<Yolov7PriceTag>,
    processor_cnn_digit: &mut Processor<CNNDigit>,
) -> String {
    let price_result: PriceResult = processor_yolo_ocr.process(crop)
        .expect("Échec de la détection des prix dans le crop");

    let corrected_group: Vec<OcrDetection> = price_result
        .detections
        .into_iter()
        .map(|mut d| {
            if is_digit_class(&d.class_label) {
                if let Ok(sub_crop) = crop_from_bbox(crop, &d.bbox) {
                    let result_cnn = processor_cnn_digit.process(&sub_crop)
                        .expect("Échec de la classification CNN sur le crop");
                    let digit_cnn = result_cnn.digit;
                    let confidence_cnn = result_cnn.confidence;
                    if d.score < confidence_cnn {
                        d.class_label = digit_cnn.to_string();
                        d.score = confidence_cnn;
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

//--------------------------------------------------------------------------------------------------

fn compare_ocr_cnn_price(
    cnn_digit_path: &str,
    yolo_v7_path: &str,
    yolo_26_price_tag_path: &str,
    device: &str,
) -> StepFn {
    let yolo_ocr_model = Yolov7PriceTag::new();
    let mut processor_yolo_ocr = Processor::new(
        yolo_ocr_model,
        yolo_v7_path,
        device,
    ).expect("Échec de la création du processeur");

    let cnn_digit_model = CNNDigit::new();
    let mut processor_cnn_digit = Processor::new(
        cnn_digit_model,
        cnn_digit_path,
        device,
    ).expect("Échec de la création du processeur");

    let yolo_price_tag_model = YOLO26::new();
    let mut processor_yolo_price_tag = Processor::new(
        yolo_price_tag_model,
        yolo_26_price_tag_path,
        device,
    ).expect("Échec de la création du processeur");

    Box::new(move |input: PipelineData| {
        let PipelineData::Detections(detections) = input else {
            panic!("expected detections at this stage")
        };

        let mut vec_detections = Arc::try_unwrap(detections)
            .unwrap_or_else(|arc| (*arc).clone());

        vec_detections.iter_mut().for_each(|detection| {
            match detection.categorie {
                DetectionClass::OtherEtiquette => {
                    // Pas de localisation : on OCR directement toute l'image,
                    // comme le box=[0,0,w,h] du Python
                    let price_str = compute_price_on_crop(
                        &detection.mask.mat,
                        &mut processor_yolo_ocr,
                        &mut processor_cnn_digit,
                    );
                    detection.price = safe_float(&price_str);
                    detection.categorie = DetectionClass::Etiquette;
                }
                DetectionClass::Etiquette => {
                    let candidate_boxes = processor_yolo_price_tag.process(&detection.mask.mat)
                    .expect("Échec de la détection des étiquettes de prix dans l'image");

                    let crops: Vec<Mat> = if candidate_boxes.is_empty() {
                        vec![detection.mask.mat.try_clone().expect("clone mat")]
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
                        let price_str = compute_price_on_crop(
                            crop,
                            &mut processor_yolo_ocr,
                            &mut processor_cnn_digit,
                        );
                        let value = safe_float(&price_str);
                        if value > best_price_val {
                            best_price_val = value;
                            best_price_str = price_str;
                        }
                    }

                    detection.price = safe_float(&best_price_str);
                }
                _ => {}
            }
        });

        processor_yolo_ocr.unload();
        processor_cnn_digit.unload();
        processor_yolo_price_tag.unload();

        PipelineData::Detections(Arc::new(vec_detections))
    })
}

//--------------------------------------------------------------------------------------------------

fn get_sam2(
    sam2_path_encoder: &str,
    sam2_path_decoder: &str,
    image: &Mat,
    device: &str,
)-> StepFn {
    let mut sam2_processor = Sam2Processor::new();
    sam2_processor.setup_model(
        sam2_path_encoder,
        sam2_path_decoder,
        device,
    );

    sam2_processor.set_image(image);

    Box::new(move |input: PipelineData| {
        let PipelineData::Detections(detections) = input else {
            panic!("expected detections at this stage")
        };

        let mut vec_detections = Arc::try_unwrap(detections)
            .unwrap_or_else(|arc| (*arc).clone());
        
        vec_detections.iter_mut().for_each(|detection| {
            match detection.categorie {
                DetectionClass::Produit => {
                    let mask = sam2_processor.predict_box(detection.bbox.xyxyn())
                    .expect("Échec de la prédiction du masque SAM2 pour le produit");
                    detection.mask._mat_bin = mask._mat_bin;
                }
                DetectionClass::Reference => {
                    let mask = sam2_processor.predict_box(detection.bbox.xyxyn())
                    .expect("Échec de la prédiction du masque SAM2 pour la référence");
                    detection.mask._mat_bin = mask._mat_bin;
                }
                _ => {}
            }
        });

        sam2_processor.unload();
        PipelineData::Detections(Arc::new(vec_detections))
    })
}

//--------------------------------------------------------------------------------------------------

/// Retourne : top-left, top-right, bottom-right, bottom-left
fn order_points(pts: &[(f32, f32)]) -> Vec<(f32, f32)> {
    let n = pts.len() as f32;
    let cx = pts.iter().map(|p| p.0).sum::<f32>() / n;
    let cy = pts.iter().map(|p| p.1).sum::<f32>() / n;

    let mut sorted: Vec<(f32, (f32, f32))> = pts
        .iter()
        .map(|&(x, y)| ((y - cy).atan2(x - cx), (x, y)))
        .collect();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let sorted: Vec<(f32, f32)> = sorted.into_iter().map(|(_, p)| p).collect();

    let start = sorted
        .iter()
        .enumerate()
        .min_by(|a, b| (a.1.0 + a.1.1).partial_cmp(&(b.1.0 + b.1.1)).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    (0..sorted.len()).map(|i| sorted[(start + i) % sorted.len()]).collect()
}

//--------------------------------------------------------------------------------------------------

pub fn compute_homography_from_reference(
    ref_mask: &Mask,
    ref_real_cm: f32,
) -> anyhow::Result<Option<(Mat, f32)>> {
    let corners_opt = ref_mask.get_reference_corners_mat()?;
    let corners = match corners_opt {
        Some(c) if c.len() == 4 => c,
        _ => return Ok(None),
    };

    let src = order_points(&corners);

    let dist = |a: (f32, f32), b: (f32, f32)| ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt();

    let side1 = dist(src[0], src[1]);
    let side2 = dist(src[1], src[2]);
    let side3 = dist(src[2], src[3]);
    let side4 = dist(src[3], src[0]);
    let ref_px = (side1 + side2 + side3 + side4) / 4.0;

    let cm_per_pixel = ref_real_cm / ref_px;

    let vec_x = ((src[1].0 - src[0].0) / side1, (src[1].1 - src[0].1) / side1);
    let vec_y = ((src[3].0 - src[0].0) / side4, (src[3].1 - src[0].1) / side4);

    let h_img = ref_mask.mat.rows() as f32;
    let w_img = ref_mask.mat.cols() as f32;
    let origin = src[0];

    let project_point = |pt: (f32, f32)| -> Point2f {
        let delta = (pt.0 - origin.0, pt.1 - origin.1);
        let x_cm = (delta.0 * vec_x.0 + delta.1 * vec_x.1) * cm_per_pixel;
        let y_cm = (delta.0 * vec_y.0 + delta.1 * vec_y.1) * cm_per_pixel;
        Point2f::new(x_cm, y_cm)
    };

    let src_corners: Vector<Point2f> = Vector::from_iter([
        Point2f::new(0.0, 0.0),
        Point2f::new(w_img, 0.0),
        Point2f::new(w_img, h_img),
        Point2f::new(0.0, h_img),
    ]);

    let dst_corners: Vector<Point2f> = Vector::from_iter([
        project_point((0.0, 0.0)),
        project_point((w_img, 0.0)),
        project_point((w_img, h_img)),
        project_point((0.0, h_img)),
    ]);

    let h = opencv::imgproc::get_perspective_transform(
        &src_corners,
        &dst_corners,
        opencv::core::DECOMP_LU,
    )?;

    Ok(Some((h, cm_per_pixel)))
}

//--------------------------------------------------------------------------------------------------

fn get_real_size_of_objects() -> StepFn {
    Box::new(move |input: PipelineData| {
        let PipelineData::Detections(detections) = input else {
            panic!("expected detections at this stage")
        };

        let mut vec_detections = Arc::try_unwrap(detections)
            .unwrap_or_else(|arc| (*arc).clone());

        // équivalent de : max_ref = max(ref_list, key=lambda x: x.score)
        let max_ref = vec_detections
            .iter()
            .filter(|d| d.categorie == DetectionClass::Reference)
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
            .cloned();

        if let Some(ref_detection) = max_ref {
            if let Ok(Some((h, _cm_per_pixel))) =
                compute_homography_from_reference(&ref_detection.mask, 10.0)
            {
                vec_detections.iter_mut().for_each(|detection| {
                    if detection.categorie != DetectionClass::Produit {
                        return;
                    }
                    if let Ok((width_cm, height_cm)) =
                        detection.get_real_size_from_homography(&h)
                    {
                        detection.set_size(width_cm, height_cm);
                    }
                });
            }
        }

        PipelineData::Detections(Arc::new(vec_detections))
    })
}

//--------------------------------------------------------------------------------------------------

fn get_real_label_of_objects() -> StepFn {
    Box::new(move |input: PipelineData| {
        let PipelineData::Detections(detections) = input else {
            panic!("expected detections at this stage")
        };

        let mut vec_detections = Arc::try_unwrap(detections)
            .unwrap_or_else(|arc| (*arc).clone());

        let has_ref = vec_detections
            .iter()
            .any(|d| d.categorie == DetectionClass::Reference);

        if has_ref {
            match ConnecteurServer::connect() {
                Ok(mut connecteur) => {
                    vec_detections.iter_mut().for_each(|detection| {
                        if detection.categorie != DetectionClass::Produit {
                            return;
                        }

                        let family_id = match connecteur.get_family_id(&detection.label) {
                            Ok(Some(id)) => id,
                            _ => return,
                        };

                        let size_info = match connecteur.get_size_of_products(family_id) {
                            Ok(list) if !list.is_empty() => list,
                            _ => return,
                        };

                        let height_computed = detection.height_cm as f64;
                        let mut distance_to_height = 100.0_f64;

                        for (ean, height, _largeur) in size_info {
                            let distance = (height_computed - height).abs();
                            if distance < distance_to_height {
                                distance_to_height = distance;
                                detection.label = ean;
                            }
                        }
                    });
                }
                Err(e) => {
                    eprintln!("Erreur de connexion à la base : {e}");
                }
            }
        }

        PipelineData::Detections(Arc::new(vec_detections))
    })
}

//--------------------------------------------------------------------------------------------------
// End of file
//--------------------------------------------------------------------------------------------------