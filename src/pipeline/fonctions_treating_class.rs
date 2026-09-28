//--------------------------------------------------------------------------------------------------
// Filename: fonctions_treating_class.rs
// Author: Jean Anquetil
// Date: 2026-09-28
//--------------------------------------------------------------------------------------------------

use crate::detections::detections::{Detection, DetectionClass};

//--------------------------------------------------------------------------------------------------

pub fn treat_product_detection(
    detection: &mut Detection,
    h: &Mat
) -> &mut Detection {
    let result_inception =
        processor_inception.process(&detection.mask.mat)
        .expect("Échec de la classification InceptionV3");
    let result_efficient = 
        processor_efficientnet.process(&detection.mask.mat)
        .expect("Échec de la classification EfficientNetB2");
    let mask = sam2_processor.predict_box(detection.bbox.xyxyn())
        .expect("Échec de la prédiction du masque SAM2 pour le produit");
    
    detection.mask._mat_bin = mask._mat_bin;
    
    let label_inception = result_inception.class_label;
    let score_inception = result_inception.score;

    let label_efficient = result_efficient.class_label;
    let score_efficient = result_efficient.score;
    let energy_efficient = result_efficient.energy;

    if score_inception >= 0.996 {
        detection.label = label_inception;
        detection.score = score_inception;
    } else if score_inception >= 0.57 && score_inception < 0.996 {
        detection.label = if score_efficient >= 0.23 {
            label_efficient
        } else {
            "OOD".to_string()
        };
        detection.score = score_efficient;
    } else {
        detection.label = "OOD".to_string();
        detection.score = 0.0;
        if energy_efficient < -5.0 as f32 && conf_efficientnet > 0.5 as f32 {
            detection.label = label_efficient;
            detection.score = score_efficient;
        }
    }

    let Ok((width_cm, height_cm)) = detection.get_real_size_from_homography(&h);
    detection.set_size(width_cm, height_cm);

    return detection;
}

//--------------------------------------------------------------------------------------------------

pub fn treat_etiquette(
    detection: &mut Detection,
) -> &mut Detection {
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
    return detection;
}

//--------------------------------------------------------------------------------------------------

pub fn treat_publicity(
    detection : &mut Detection,
) -> &mut Detection {
    

    let new_results = processor_yolo_price_tag.process(&detection.mask.mat)
        .expect("Échec de la détection des étiquettes dans les publicités");

    let mut best_score = 0.0;
    let mut best_detection = None;

    new_results.iter().for_each(|new_detection| {
        if new_detection.categorie != DetectionClass::Price {
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
        best.categorie = DetectionClass::Etiquette;

        best = treat_etiquette(&mut best);
    }

    return ok((best, detection));
}

//--------------------------------------------------------------------------------------------------
// End of file
//--------------------------------------------------------------------------------------------------