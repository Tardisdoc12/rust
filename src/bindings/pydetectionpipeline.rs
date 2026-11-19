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
use serde::ser::StdError;

use crate::processor::processor::Processor;
use crate::models::model_core::ModelPipeline;
use crate::models::{
    tiled_yolo_26::{TiledYOLO26, TiledConfig},
    sam2::Sam2Processor,
    qwen_onnx::QwenOnnx,
    vlm::Vlm,
    identifier::{Gate, Identifier, Route},
    florence::Florence,
};
use crate::detections::{mask::Mask, detection_class::DetectionClass, detections::Detection};

use crate::functions_::positionnement::set_all_position;
use crate::functions_::tiled_merge::MergeConfig;
use crate::tools_class::config::{lire_config, Config};
use crate::tools_class::label_resolver::LabelResolver;
use crate::functions_::utils::compute_homography_from_reference;

//--------------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

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

fn dist2(a: (f32, f32), b: (f32, f32)) -> f32 {
    let (dx, dy) = (a.0 - b.0, a.1 - b.1);
    dx * dx + dy * dy
}

fn nearest_object(objects: &[(usize, (f32, f32))], center: (f32, f32)) -> Option<usize> {
    objects
        .iter()
        .min_by(|a, b| dist2(a.1, center).total_cmp(&dist2(b.1, center)))
        .map(|&(i, _)| i)
}

const REFERENCE_REAL_CM: f32 = 10.0;

//--------------------------------------------------------------------------------------------------

#[pyclass(name = "DetectionPipeline")]
pub struct PyDetectionPipeline {
    config: Config,
    processor_yolo: Mutex<Processor<TiledYOLO26>>,
    identifier: Mutex<Identifier>,
    number_photos_candidates_for_vlm: usize,
    vlm: Mutex<Vlm>,
    sam2_processor: Mutex<Sam2Processor>,
    florence: Mutex<Processor<Florence>>,
    resolver: LabelResolver,
}

#[pymethods]
impl PyDetectionPipeline {
    #[new]
    fn new(
        config_path: &str
    ) -> PyResult<Self> {
        let config = lire_config(config_path).map_err(|arg0: std::boxed::Box<dyn StdError>| to_py_err(/* anyhow::Error */))?;

        let resolver = LabelResolver::load().map_err(to_py_err)?;
        // Chargement des modèles EN PARALLÈLE (fichiers et sessions ONNX
        // indépendants, aucune synchronisation nécessaire entre eux).
        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("ort=warn"));
        let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
        let result: anyhow::Result<_> = thread::scope(|scope| {
            let h_yolo = scope.spawn(|| Processor::new(TiledYOLO26::with_config(TiledConfig {
                overlap: 0.3,
                conf: 0.30,
                scales: vec![1.0, 0.5],
                merge: MergeConfig { class_groups: vec![vec![1, 2, 3, 4]], ..Default::default() },
                ..Default::default()
            }), &config.yolo_26_path, &config.device));
            let h_sam2 = scope.spawn(|| -> anyhow::Result<Sam2Processor> {
                let mut sam2 = Sam2Processor::new();
                sam2.setup_model(&config.sam2_path_encoder, &config.sam2_path_decoder, &config.device)?;
                Ok(sam2)
            });
            let h_florence = scope.spawn(|| Processor::new(Florence::new(), &config.florence_path, &config.device));

            let h_embbed = scope.spawn(|| -> anyhow::Result<Identifier> {
                let mut ident = Identifier::new(&config.identifier_base_path, config.number_candidate)?;          // dossier avec meta.json + embeddings.npy
                ident.setup_model(&config.embedder_path, &config.device)?;
                Ok(ident)
            });

            let h_backend = scope.spawn(|| -> anyhow::Result<Vlm> {
                let backend = QwenOnnx::new(
                    &config.visual_path, &config.embed_path, &config.decoder_path,
                    &config.tokenizer_json, config.kv_heads, config.head_dim, config.tokens_per_second,
                    &config.device)?;
                let mut vlm = Vlm::new(Box::new(backend));
                vlm.perms = config.perms;
                vlm.early_conf = Some(config.early_conf);
                Ok(vlm)
            });

            Ok((
                h_yolo.join().map_err(|_| anyhow::anyhow!("thread yolo a paniqué"))??,
                h_sam2.join().map_err(|_| anyhow::anyhow!("thread sam2 a paniqué"))??,
                h_backend.join().map_err(|_| anyhow::anyhow!("thread backend a paniqué"))??,
                h_embbed.join().map_err(|_| anyhow::anyhow!("thread embedder a paniqué"))??,
                h_florence.join().map_err(|_| anyhow::anyhow!("thread florence a paniqué"))??,
            ))
        });

        let (
            processor_yolo,
            sam2_processor,
            backend,
            identifier,
            florence
        ) = result.map_err(to_py_err)?;

        Ok(Self {
            processor_yolo: Mutex::new(processor_yolo),
            sam2_processor: Mutex::new(sam2_processor),
            vlm: Mutex::new(backend),
            identifier: Mutex::new(identifier),
            florence: Mutex::new(florence),
            config,
            number_photos_candidates_for_vlm: config.number_photos_candidates_for_vlm,
            resolver,
        })
    }

    fn unload(&self) {
        let _ = self.processor_yolo.lock().map(|mut p| p.unload());
        let _ = self.sam2_processor.lock().map(|mut p| p.unload());
        let _ = self.identifier.lock().map(|mut i| i.unload());
        let _ = self.florence.lock().map(|mut f| f.unload());
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
        let mut objects: Vec<(usize, (f32, f32))> = Vec::new();
        let mut number_etiquette: i8 = 0;
        let mut has_ref : bool = false;

        for (i, detection) in vec_detections.iter_mut().enumerate() {
            match detection.categorie {
                DetectionClass::Reference => {
                    if let Ok(Some((h, _))) = self.classify_reference(detection) {
                        homography = Some(h);
                    }
                    has_ref = true;
                }
                DetectionClass::Etiquette => {
                    number_etiquette += 1;
                    detection.label = "PL".to_string();
                    detection.master_product_id = self.resolver.get_master_product_id(&detection.label).unwrap_or_default();
                }
                DetectionClass::Publicity => {
                    detection.label = "AD".to_string();
                    detection.master_product_id = self.resolver.get_master_product_id(&detection.label).unwrap_or_default();
                }
                DetectionClass::NoProduct => {
                    detection.label = "ES".to_string();
                    objects.push((i, detection.bbox.center_rel()));
                    detection.master_product_id = self.resolver.get_master_product_id(&detection.label).unwrap_or_default();
                }
                _ => objects.push((i, detection.bbox.center_rel())),
            }
        }
        if has_ref{
            let _ = self.sam2_processor.lock().unwrap().set_image(&image_rust.mat);
        }

        let mut new_etiquettes: Vec<Detection> = Vec::new();
        let mut assignments: Vec<(usize, f32)> = Vec::new();
        for detection in vec_detections.iter_mut() {
            match detection.categorie {
                DetectionClass::Produit => {
                    self.classify_produit(detection, homography.as_ref(), has_ref);
                    self.resolver.resolve(detection);
                }
                DetectionClass::Etiquette => {
                    self.classify_etiquette(detection);
                    if let Some(idx) = nearest_object(&objects, detection.bbox.center_rel()) {
                        assignments.push((idx, detection.price));
                    }
                }
                DetectionClass::Publicity => {
                    self.classify_publicity(detection);
                    if detection.price > 0.0 {
                        number_etiquette += 1;
                        if let Some(idx) = nearest_object(&objects, detection.bbox.center_rel()) {
                            assignments.push((idx, detection.price));
                        }
                    }
                }
                _ => {}
            }
        }

        for (idx, price) in assignments {
            vec_detections[idx].price = price;
        }
        vec_detections.extend(new_etiquettes);
        if number_etiquette >= 2{
            set_all_position(&mut vec_detections);
        }
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

    fn classify_produit(&self, detection: &mut Detection, h: Option<&Mat>, has_ref: bool) ->  Result<(), Box<dyn std::error::Error>> {
        let gate = Gate { 
            accept_sim: self.config.accept_sim,
            accept_margin: self.config.accept_margin,
            reject_sim: self.config.reject_sim,
            cand_margin: self.config.cand_margin
        };
        match self.identifier.lock().unwrap().route(&detection.mask.mat, &gate, self.number_photos_candidates_for_vlm)? {
            Route::Accept { ean, name, sim, margin } => {
                detection.label = ean.clone();
                detection.score = sim;
            },
            Route::Reject { sim } => {
                detection.label = "OOB".to_string();
                detection.score = 1.0 - sim;
            },
            Route::Ask { cands, sim, margin } => {
                let r = self.vlm.lock().unwrap().process_product(&detection.mask.mat, &cands)?;
                match &r.ean {
                    Some(ean) => {
                        detection.label = ean.clone();
                        detection.score = r.conf;
                        println!("VLM : {ean} (conf {:.2}, ordres d'accord {}) [sim {sim:.3}, écart {margin:.3}]", r.conf, r.agree);
                    },
                    None => {
                        detection.label = "OOB".to_string();
                        detection.score = 1.0 - r.conf;
                        println!("VLM : aucun des candidats (OOB {:.2})", r.conf);
                    },
                }
            }
        }

        if !has_ref {
            return Ok(());
        }

        match self.predict_mask(detection.bbox.xyxyn()) {
            Ok(mask) => detection.mask._mat_bin = mask._mat_bin,
            Err(e) => eprintln!("[produit] sam2: {e:#}"),
        }
        // Taille réelle : seulement si on a une homographie
        if let Some(h) = h {
            if let Ok((width_cm, height_cm)) = detection.get_real_size_from_homography(h) {
                detection.set_size(width_cm, height_cm);
            }
        }
    }

    fn classify_etiquette(&self, detection: &mut Detection) {
        let Ok(result) = process_locked(&self.florence, &detection.mask.mat) else { return };
        let best_price_f32 = result.price as f32;
        if best_price_f32.is_none() {
            detection.price = 0.0 as f32;
            return Ok(());
        }
        detection.price = best_price_f32.unwrap();
    }

    /// Renvoie l'étiquette (prix) trouvée dans la publicité, si elle existe.
    fn classify_publicity(
        &self,
        detection: &Detection,
    ) -> anyhow::Result<Option<Detection>> {
        let Ok(result) = process_locked(&self.florence, &detection.mask.mat) else { return };
        let best_price_f32 = result.price as f32;
        if best_price_f32.is_none() {
            detection.price = 0.0 as f32;
            return Ok(());
        }
        detection.price = best_price_f32.unwrap();
    }

    fn classify_reference(&self, detection: &mut Detection) -> anyhow::Result<Option<(Mat, f32)>> {
        let mask = self.predict_mask(detection.bbox.xyxyn())?;
        detection.mask._mat_bin = mask._mat_bin;
        compute_homography_from_reference(&detection.mask, REFERENCE_REAL_CM)
    }
}

//--------------------------------------------------------------------------------------------------
// End of file
//--------------------------------------------------------------------------------------------------