mod models;
mod processor;
mod functions_;
use models::inceptionv3::InceptionV3;
use models::efficientnetb2::EfficientNetB2;
use models::yolo_v7::Yolov7PriceTag;
use models::yolo_26::YOLO26;
use models::cnn_digit::CNNDigit;
use models::sam2::Sam2Processor;
use processor::processor::Processor;
use std::time::Instant;
use opencv::prelude::*;

fn main()  -> anyhow::Result<()>{
    // --- Chronométrage du chargement des modèles ---
    let start_load_inception = Instant::now();
    let inception_model = InceptionV3::new();
    let mut processor = Processor::new(
        inception_model,
        "C:\\Users\\laure\\Desktop\\rust\\inception_v3.onnx",
        "cuda",
    ).expect("Échec de la création du processeur");
    let duration_load_inception = start_load_inception.elapsed();

    let yolo_ocr_model = Yolov7PriceTag::new();
    let mut processor_yolo_ocr = Processor::new(
        yolo_ocr_model,
        "C:\\Users\\laure\\Desktop\\rust\\yolo_v7.onnx",
        "cuda",
    ).expect("Échec de la création du processeur");

    let cnn_digit_model = CNNDigit::new();
    let mut processor_cnn_digit = Processor::new(
        cnn_digit_model,
        "C:\\Users\\laure\\Desktop\\rust\\digit_cnn.onnx",
        "cuda",
    ).expect("Échec de la création du processeur");

    let start_load_yolo = Instant::now();
    let yolo_model = YOLO26::new();
    let mut processor_yolo = Processor::new(
        yolo_model,
        "C:\\Users\\laure\\Desktop\\rust\\yolo_26.onnx",
        "cuda",
    ).expect("Échec de la création du processeur");
    let duration_load_yolo = start_load_yolo.elapsed();

    let start_load_sam2 = Instant::now();
    let mut sam2_processor = Sam2Processor::new();
    sam2_processor.setup_model(
        "C:\\Users\\laure\\Desktop\\rust\\sam2_encoder.onnx",
        "C:\\Users\\laure\\Desktop\\rust\\sam2_decoder.onnx",
        "cuda",
    )?;
    let duration_load_sam2 = start_load_sam2.elapsed();

    let start_load_effnet = Instant::now();
    let efficientnet_b2 = EfficientNetB2::new();
    let mut processor_b2 = Processor::new(
        efficientnet_b2,
        "C:\\Users\\laure\\Desktop\\rust\\efficientnet_b2.onnx",
        "cuda",
    ).expect("Échec de la création du processeur");
    let duration_load_effnet = start_load_effnet.elapsed();

    println!("Chargement InceptionV3   : {:.3?}", duration_load_inception);
    println!("Chargement EfficientNetB2: {:.3?}", duration_load_effnet);
    println!("Chargement YOLO26         : {:.3?}", duration_load_yolo);
    println!("Chargement SAM2           : {:.3?}", duration_load_sam2);

    // --- Chargement de l'image (hors chrono d'inférence) ---
    let image_path = "C:\\Users\\laure\\Desktop\\Dataset_original\\Dataset_original\\12111102009\\20260424_143123_20260424_143123-frame_0001-2.jpg";
    
    let image_path_2 = "C:\\Users\\laure\\Downloads\\photos_a_annoter\\03_CQ00027_00_05_003_07_2026-08-18_17-25.jpg";
    let image_path_3 = "C:\\Users\\laure\\Desktop\\oil_eduardo\\etiquette_real.png";

    let img = opencv::imgcodecs::imread(image_path, opencv::imgcodecs::IMREAD_COLOR)
        .expect("Échec du chargement de l'image");

    let img_2 = opencv::imgcodecs::imread(image_path_2, opencv::imgcodecs::IMREAD_COLOR)
        .expect("Échec du chargement de l'image");

    let img_3 = opencv::imgcodecs::imread(image_path_3, opencv::imgcodecs::IMREAD_COLOR)
        .expect("Échec du chargement de l'image");

    sam2_processor.set_image(&img)?;

    // --- Chronométrage de l'inférence ---
    let start_infer_inception = Instant::now();
    let output = processor.process(&img);
    let duration_infer_inception = start_infer_inception.elapsed();

    let start_infer_effnet = Instant::now();
    let output_b2 = processor_b2.process(&img);
    let duration_infer_effnet = start_infer_effnet.elapsed();

    let start_infer_yolo = Instant::now();
    let output_yolo = processor_yolo.process(&img_2)?;
    let duration_infer_yolo = start_infer_yolo.elapsed();

    let output_yolo_v7 = processor_yolo_ocr.process(&img_3)?;
    let output_cnn_digit = processor_cnn_digit.process(&img_3)?;

    let sam2_mask_result = sam2_processor.predict_box(
        (0.0, 0.0, img.cols() as f32, img.rows() as f32),
    )?;

    println!("\nInférence InceptionV3   : {:.3?}", duration_infer_inception);
    println!("Inférence EfficientNetB2: {:.3?}", duration_infer_effnet);
    println!("Inférence YOLO26         : {:.3?}", duration_infer_yolo);
    println!("Total des trois inférences: {:.3?}", duration_infer_inception + duration_infer_effnet + duration_infer_yolo);


    // --- Résultats ---
    println!("\nOutput InceptionV3: {:?}", output);
    println!("Output EfficientNetB2: {:?}", output_b2);
    println!("Prix détecté : {}", output_yolo_v7.price_str);
    println!("Chiffres détectés : {:?}", output_cnn_digit);
    println!("Masque SAM2 : {}x{}, score: {:.3}", sam2_mask_result.width, sam2_mask_result.height, sam2_mask_result.score);

    for detection in output_yolo.detections {
        println!(
            "{} {:.2} [{:.0}, {:.0}, {:.0}, {:.0}]",
            detection.class_label,
            detection.score,
            detection.bbox.x1,
            detection.bbox.y1,
            detection.bbox.x2,
            detection.bbox.y2,
        );
    }

    
    Ok(())
}