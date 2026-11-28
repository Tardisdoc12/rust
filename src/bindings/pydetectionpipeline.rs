//--------------------------------------------------------------------------------------------------
// Filename: bindings/pydetectionpipeline.rs
//--------------------------------------------------------------------------------------------------

use std::sync::Mutex;
use std::thread;

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use numpy::PyReadonlyArray3;
use opencv::core::Mat;
use opencv::prelude::MatTraitConst;
use tracing_subscriber::EnvFilter;

use crate::processor::processor::Processor;
use crate::models::model_core::ModelPipeline;
use crate::models::{
    tiled_yolo_26::{TiledYOLO26, TiledConfig},
    sam2::Sam2Processor,
    qwen_onnx::QwenOnnx,
    vlm::{Vlm, OOB},
    identifier::{Decision, Gate, Identifier, Route},
    florence::Florence,
};
use crate::detections::{mask::Mask, detection_class::DetectionClass, detections::Detection};

use crate::functions_::positionnement::set_all_position;
use crate::functions_::tiled_merge::MergeConfig;
use crate::tools_class::config::{lire_config, Config};
use crate::tools_class::label_resolver::LabelResolver;
use crate::functions_::utils::compute_homography_from_reference;

//--------------------------------------------------------------------------------------------------

fn to_py_err(e: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{e:#}"))
}

fn process_locked<M: ModelPipeline>(
    processor: &Mutex<Processor<M>>,
    image: &Mat,
) -> anyhow::Result<M::Output> {
    processor
        .lock()
        .map_err(|_| anyhow::anyhow!("Mutex empoisonné (un thread a paniqué en tenant le verrou)"))?
        .process(image)
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
        let config = lire_config(config_path).map_err(|e| PyRuntimeError::new_err(format!("config {config_path}: {e}")))?;

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
            config: config.clone(),
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

    #[pyo3(signature = (image, is_pallet=false))]
    fn process(&self, image: PyReadonlyArray3<'_, u8>, is_pallet: bool) -> PyResult<Vec<Detection>> {
        let mut image_rust = Mask::new();
        image_rust.setup_from_python(image)?;
       
        // --- Étape 1 : détection principale
        
        let mut vec_detections = process_locked(&self.processor_yolo, &image_rust.mat)
            .map_err(to_py_err)?;

        let mut homography: Option<Mat> = None;
        let mut objects: Vec<(usize, (f32, f32))> = Vec::new();
        let mut number_etiquette: i8 = 0;
        let mut has_ref : bool = false;
        // Palette (UN SEUL produit) : avec au moins 2 produits, on ne les classe pas un par un. On pool leurs embeddings
        // et on décide une seule fois pour toute la palette. Un seul produit : comportement normal.
        let pallet_mode = is_pallet
            && vec_detections.iter().filter(|d| matches!(d.categorie, DetectionClass::Produit)).count() >= 2;
        let mut pallet_owners: Vec<usize> = Vec::new();    // indice de la détection dans vec_detections
        let mut pallet_bests: Vec<Vec<f32>> = Vec::new();  // sa meilleure similarité avec chaque EAN

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
            let idx: Vec<usize> = vec_detections.iter().enumerate()
                .filter(|(_, d)| matches!(d.categorie, DetectionClass::Produit)).map(|(i, _)| i).collect();
                    let boxes: Vec<_> = idx.iter().map(|&i| vec_detections[i].bbox.xyxyn()).collect();
                    match self.sam2_processor.lock().unwrap().predict_boxes(&boxes) {
                        Ok(masks) => for (&i, m) in idx.iter().zip(masks) {
                            vec_detections[i].mask._mat_bin = m._mat_bin;
                            if let Some(h) = homography.as_ref() {
                                if let Ok((w, hh)) = vec_detections[i].get_real_size_from_homography(h) {
                                    vec_detections[i].set_size(w, hh);
                                }
                            }
                        },
                        Err(e) => eprintln!("[produit] sam2: {e:#}"),
            }
        }

        let mut assignments: Vec<(usize, f32)> = Vec::new();
        for (det_i, detection) in vec_detections.iter_mut().enumerate() {
            match detection.categorie {
                DetectionClass::Produit => {
                    if pallet_mode {
                        // palette : on garde seulement les similarités ; la décision vient après la boucle
                        match self.embed_best(detection) {
                            Ok(b) => { pallet_owners.push(det_i); pallet_bests.push(b); }
                            Err(e) => eprintln!("[palette] embedding impossible : {e:#}"),
                        }
                    } else {
                        self.classify_produit_or_oob(detection);
                    }
                }
                DetectionClass::Etiquette => {
                    if let Err(e) = self.classify_etiquette(detection) {
                        eprintln!("[etiquette] {e:#}");
                    }
                    if let Some(idx) = nearest_object(&objects, detection.bbox.center_rel()) {
                        assignments.push((idx, detection.price));
                    }
                }
                DetectionClass::Publicity => {
                    if let Err(e) = self.classify_publicity(detection) {
                        eprintln!("[publicity] {e:#}");
                    }
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

        if pallet_mode {
            if pallet_bests.is_empty() {
                // aucun embedding exploitable : on retombe sur la classification détection par détection
                for d in vec_detections.iter_mut().filter(|d| matches!(d.categorie, DetectionClass::Produit)) {
                    self.classify_produit_or_oob(d);
                }
            } else {
                self.resolve_pallet(&mut vec_detections, &pallet_owners, &pallet_bests);
            }
        }

        for (idx, price) in assignments {
            vec_detections[idx].price = price;
        }
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

    fn classify_produit(&self, detection: &mut Detection) ->  Result<(), Box<dyn std::error::Error>> {
        let gate = Gate { 
            accept_sim: self.config.accept_sim,
            accept_margin: self.config.accept_margin,
            reject_sim: self.config.reject_sim,
            cand_margin: self.config.cand_margin
        };
        match self.identifier.lock().unwrap().route(&detection.mask.mat, &gate, self.number_photos_candidates_for_vlm)? {
            Route::Accept { ean, name: _, sim, .. } => {
                detection.label = ean.clone();
                detection.score = sim;
            },
            Route::Reject { sim } => {
                detection.label = "OOB".to_string();
                detection.score = 1.0 - sim;
            },
            Route::Ask { cands, sims, sim, margin } => {
                let r = self.vlm.lock().unwrap().process_product(&detection.mask.mat, &cands)?;
                let d = Decision::default();
                let agree_ok = r.agree || !d.require_agree || r.n_ok < 2;

                // 1) VLM confiant et cohérent : on le suit (comme en Python)
                if r.conf >= d.min_conf && agree_ok {
                    match &r.ean {
                        Some(ean) => {
                            detection.label = ean.clone();
                            detection.score = r.conf;
                            println!("VLM : {ean} (conf {:.2}, ordres d'accord {}) [sim {sim:.3}, écart {margin:.3}]", r.conf, r.agree);
                        }
                        None => {
                            detection.label = "OOB".to_string();
                            detection.score = r.conf;
                            println!("VLM : aucun des candidats (OOB {:.2})", r.conf);
                        }
                    }
                    return Ok(());
                }

                let why = if !agree_ok { "VLM incohérent entre ordres".to_string() } else { format!("VLM peu confiant ({:.2})", r.conf) };

                // 2) VLM hésitant, pas de fusion : on garde l'embedding top-1
                if d.fuse_w <= 0.0 {
                    detection.label = cands[0].ean.clone();
                    detection.score = sim;
                    println!("VLM incertain ({why}) : embedding conservé {} [sim {sim:.3}]", detection.label);
                    return Ok(());
                }

                // 3) fusion VLM x embedding : score(c) = P_vlm(c) * exp(w * (sim(c) - sim_top1) / T)
                let mut fused: Vec<(String, f32)> = cands.iter().zip(&sims).map(|(c, &s)| {
                    let p = r.probs.get(&c.ean).copied().unwrap_or(0.0);
                    (c.ean.clone(), p * (d.fuse_w * (s - sim) / d.fuse_t).exp())
                }).collect();
                if let Some(&p_oob) = r.probs.get(OOB) {
                    let prior = match d.oob_sim {
                        Some(o) => (d.fuse_w * (o - sim) / d.fuse_t).exp(),
                        None => d.none_prior,
                    };
                    fused.push((OOB.to_string(), p_oob * prior));
                }
                let tot = fused.iter().map(|x| x.1).sum::<f32>();
                if !(tot > 0.0 && tot.is_finite()) {            // dégénéré : on garde l'embedding
                    detection.label = cands[0].ean.clone();
                    detection.score = sim;
                    return Ok(());
                }
                let (w, p) = fused.iter().max_by(|a, b| a.1.total_cmp(&b.1)).map(|(k, v)| (k.clone(), v / tot)).unwrap();
                detection.label = w.clone();
                detection.score = p;
                println!("VLM fusion : {w} ({p:.2}) [{why} ; sim {sim:.3}, écart {margin:.3}] {}",
                    fused.iter().map(|(k, v)| format!("{k}: {:.2}", v / tot)).collect::<Vec<_>>().join(", "));
            }
        }
        Ok(())
    }

    /// Classification détection par détection, avec repli OOB (jamais de "Product" en base).
    fn classify_produit_or_oob(&self, detection: &mut Detection) {
        if let Err(e) = self.classify_produit(detection) {
            eprintln!("[produit] classification échouée : {e}");
            detection.label = "OOB".to_string();
            detection.score = 0.0;
        }
        self.resolver.resolve(detection);
    }

    /// Palette : meilleure similarité du crop avec chaque EAN de la base (index = code EAN de Db).
    fn embed_best(&self, detection: &Detection) -> anyhow::Result<Vec<f32>> {
        use crate::models::model_core::ModelPipeline;
        let m = &detection.mask.mat;
        if m.empty() {
            anyhow::bail!("crop vide ({}x{})", m.cols(), m.rows());
        }
        let mut id = self.identifier.lock().map_err(|_| anyhow::anyhow!("Mutex Identifier empoisonné"))?;
        let x = id.preprocess(m)?;
        id.infer(&x)
    }

    /// Palette = un seul produit : une décision pour toutes les détections `Produit`
    /// (label, score et taille identiques). Voir pallet.rs pour la règle.
    fn resolve_pallet(&self, dets: &mut [Detection], owners: &[usize], bests: &[Vec<f32>]) {
        use crate::models::pallet::{decide, median_size, Verdict};
        let params = &self.config.pallet;
        let Some(verdict) = decide(bests, params) else {
            eprintln!("[palette] similarités incohérentes : classement détection par détection");
            for d in dets.iter_mut().filter(|d| matches!(d.categorie, DetectionClass::Produit)) {
                self.classify_produit_or_oob(d);
            }
            return;
        };
        let (label, score) = match verdict {
            Verdict::Accept { code, score } => {
                let ean = self.identifier.lock().unwrap().ean(code).to_string();
                println!("[palette] {} détections -> {ean} (S = {score:.3}, sans VLM)", bests.len());
                (ean, score)
            }
            Verdict::Reject { score } => {
                println!("[palette] {} détections -> OOB (S max = {score:.3}, sans VLM)", bests.len());
                (OOB.to_string(), 1.0 - score)
            }
            Verdict::Ambiguous { codes, score } => match self.arbitrate_pallet(dets, owners, bests, &codes) {
                Ok(Some((label, conf))) => {
                    println!("[palette] {} détections -> {label} (VLM, conf {conf:.2}, S = {score:.3})", bests.len());
                    (label, conf)
                }
                other => {
                    if let Err(e) = other {
                        eprintln!("[palette] arbitrage VLM impossible : {e:#}");
                    }
                    // VLM absent ou hésitant : on suit les embeddings poolés (top-1 de la palette), sauf sous reject_sim
                    if params.reject_sim.map_or(true, |t| score >= t) {
                        let ean = self.identifier.lock().unwrap().ean(codes[0]).to_string();
                        println!("[palette] ambigu, repli embeddings : {ean} (S = {score:.3})");
                        (ean, score)
                    } else {
                        println!("[palette] ambigu, repli embeddings : OOB (S = {score:.3})");
                        (OOB.to_string(), 1.0 - score)
                    }
                }
            },
        };

        let prod: Vec<usize> = dets.iter().enumerate()
            .filter(|(_, d)| matches!(d.categorie, DetectionClass::Produit)).map(|(i, _)| i).collect();
        let size = median_size(&prod.iter().map(|&i| (dets[i].width_cm, dets[i].height_cm)).collect::<Vec<_>>());
        for &i in &prod {
            let d = &mut dets[i];
            d.label = label.clone();
            d.score = score;
            if let Some((w, h)) = size {
                d.set_size(w, h);
            }
            self.resolver.resolve(d);
        }
    }

    /// Cas ambigu : UN appel VLM, sur la détection la plus proche du meilleur candidat, contre les meilleurs EAN de la
    /// palette. Renvoie None si le VLM est désactivé ou pas assez sûr.
    fn arbitrate_pallet(&self, dets: &[Detection], owners: &[usize], bests: &[Vec<f32>], codes: &[usize])
        -> anyhow::Result<Option<(String, f32)>> {
        use crate::models::pallet::best_crop;
        if !self.config.pallet.use_vlm || codes.is_empty() {
            return Ok(None);
        }
        let det = &dets[owners[best_crop(bests, codes[0])]];
        let cands = self.identifier.lock().map_err(|_| anyhow::anyhow!("Mutex Identifier empoisonné"))?
            .cands_for_crop(&det.mask.mat, codes, self.number_photos_candidates_for_vlm)?;
        let r = self.vlm.lock().map_err(|_| anyhow::anyhow!("Mutex VLM empoisonné"))?
            .process_product(&det.mask.mat, &cands)?;
        let d = Decision::default();
        let agree_ok = r.agree || !d.require_agree || r.n_ok < 2;
        if r.conf >= d.min_conf && agree_ok {
            return Ok(Some((r.ean.unwrap_or_else(|| OOB.to_string()), r.conf)));
        }
        Ok(None)
    }

    fn classify_etiquette(&self, detection: &mut Detection) -> anyhow::Result<()> {
        let m = &detection.mask.mat;
        if m.empty() {
            anyhow::bail!("crop vide ({}x{})", m.cols(), m.rows());
        }
        let result = process_locked(&self.florence, m)?;
        if result.price.is_none() {
            eprintln!("[florence] aucun prix extrait du crop {}x{}", m.cols(), m.rows());
            // si ton résultat Florence a un champ texte brut, affiche-le ici : c'est le plus parlant
        }
        detection.price = result.price.map(|p| p as f32).unwrap_or(0.0);
        Ok(())
    }

    /// Renvoie l'étiquette (prix) trouvée dans la publicité, si elle existe.
    fn classify_publicity(&self, detection: &mut Detection) -> anyhow::Result<()> {
        let m = &detection.mask.mat;
        if m.empty() {
            anyhow::bail!("crop vide ({}x{})", m.cols(), m.rows());
        }
        let result = process_locked(&self.florence, m)?;
        if result.price.is_none() {
            eprintln!("[florence] aucun prix extrait du crop {}x{}", m.cols(), m.rows());
            // si ton résultat Florence a un champ texte brut, affiche-le ici : c'est le plus parlant
        }
        detection.price = result.price.map(|p| p as f32).unwrap_or(0.0);
        Ok(())
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