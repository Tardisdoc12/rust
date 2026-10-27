//--------------------------------------------------------------------------------------------------
// Filename: tiled_yolo_26.rs
// Rôle    : détection YOLO26 (ONNX / ort) sur de très grandes images, par tuiles.
//           Port Rust de TiledYOLO (Python) dans l'architecture de yolo_26.rs.
//
// Principe :
//   1. preprocess  : calcule le plan de tuilage (échelles x tuiles) et garde l'image source.
//   2. infer       : pour chaque tuile -> crop (+ letterbox des bords) -> tenseur -> session ONNX.
//                    Les tenseurs sont construits tuile par tuile (jamais N x 3 x 640 x 640 en mémoire).
//   3. postprocess : boîtes remises dans le repère de l'image complète, flags de couture, nettoyage
//                    (voir tiled_merge.rs), puis conversion vers BigDetection comme YOLO26.
//
// Fichiers associés (même dossier `models/`) :
//   - tiled_geometry.rs : positions des tuiles, letterbox, conversion tuile -> image (pur, testé)
//   - tiled_merge.rs    : suppression / fusion des fragments, boîtes imbriquées, NMS (pur, testé)
//--------------------------------------------------------------------------------------------------

use std::str::FromStr;
use std::sync::Mutex;

use opencv::core::{Mat, Rect, Scalar, Size};
use opencv::imgproc;
use opencv::prelude::*;
use ort::session::Session;

use crate::detections::bbox::BBox;
use crate::detections::detection_class::DetectionClass;
use crate::detections::detections::Detection as BigDetection;
use crate::detections::mask::Mask;
use crate::functions_::setup_model_avec_cache::setup_model_avec_cache;
use crate::models::model_core::ModelPipeline;
use crate::functions_::tiled_geometry::{build_plan, rows_to_cands, scaled_size, TilePlan};
use crate::functions_::tiled_merge::MergeConfig;
use crate::models::yolo_26::crop_mat; // -> passer `fn crop_mat` en `pub(crate) fn crop_mat` dans yolo_26.rs

//--------------------------------------------------------------------------------------------------

/// Taille de tuile = taille d'entrée du modèle ONNX (640 : l'export est à taille fixe).
const IMAGE_SIZE: i32 = 640;
/// Couleur du letterbox (114 = standard Ultralytics, identique à l'entraînement et à yolo_26.rs).
const FILL: f64 = 114.0;

#[derive(Debug, Clone)]
pub struct TiledConfig {
    pub overlap: f64,            // recouvrement entre tuiles (0.3 = 30 %) - f64 comme en Python (voir tiled_geometry)
    pub conf: f32,               // seuil de confiance
    pub scales: Vec<f32>,        // échelles d'inférence, ex. [1.0, 0.5] : 0.5 voit les gros objets en entier
    pub center_pad: bool,        // letterbox centré (comme YOLO) ou bas/droite
    pub drop_border_boxes: bool, // marque les boîtes coupées par une couture (requis pour le nettoyage des fragments)
    pub border_margin: f32,      // tolérance (px) pour considérer qu'une boîte colle au bord de la tuile
    pub merge: MergeConfig,      // nettoyage : fragments, imbriquées, NMS, groupes de classes
}

impl Default for TiledConfig {
    fn default() -> Self {
        TiledConfig {
            overlap: 0.3,
            conf: 0.30,
            scales: vec![1.0, 0.5],
            center_pad: true,
            drop_border_boxes: true,
            border_margin: 6.0,
            merge: MergeConfig::default(),
        }
    }
}

pub struct TiledYOLO26 {
    session: Option<Session>,
    classes: Vec<String>,
    cfg: TiledConfig,
    rows_per_tile: usize,               // nombre de lignes de sortie par tuile (300 pour YOLO26 end2end)
    plan: Mutex<Vec<TilePlan>>,         // plan de tuilage de l'image courante
    source_image: Mutex<Option<Mat>>,
}

//--------------------------------------------------------------------------------------------------
// Helpers (fonctions libres : pas d'emprunt de `self` pendant que `session` est emprunté en &mut)

/// Extrait la tuile du plan (crop + letterbox gris si c'est une tuile de bord).
fn make_tile(img: &Mat, t: &TilePlan) -> anyhow::Result<Mat> {
    // try_clone() : même appel que dans crop_mat (yolo_26.rs) ; donne un Mat possédé et continu,
    // utilisable directement par copy_make_border (BoxedRef n'implémente pas Deref).
    let crop = Mat::roi(img, Rect::new(t.x0, t.y0, t.w, t.h))?.try_clone()?;
    if t.w == IMAGE_SIZE && t.h == IMAGE_SIZE {
        return Ok(crop);
    }
    let mut padded = Mat::default();
    opencv::core::copy_make_border(
        &crop, &mut padded,
        t.pad_top, IMAGE_SIZE - t.h - t.pad_top,
        t.pad_left, IMAGE_SIZE - t.w - t.pad_left,
        opencv::core::BORDER_CONSTANT,
        Scalar::new(FILL, FILL, FILL, 0.0),
    )?;
    Ok(padded)
}

/// BGR u8 (HWC) -> RGB f32 normalisé (CHW), comme `preprocess` de yolo_26.rs.
fn mat_to_tensor(bgr: &Mat) -> anyhow::Result<Vec<f32>> {
    let mut rgb = Mat::default();
    imgproc::cvt_color(bgr, &mut rgb, imgproc::COLOR_BGR2RGB, 0)?;

    let data: &[u8] = rgb.data_bytes()?;
    let hw = (rgb.rows() as usize) * (rgb.cols() as usize);

    let mut tensor = vec![0f32; 3 * hw];
    for (i, px) in data.chunks_exact(3).enumerate() {
        tensor[i] = px[0] as f32 / 255.0;
        tensor[hw + i] = px[1] as f32 / 255.0;
        tensor[2 * hw + i] = px[2] as f32 / 255.0;
    }
    Ok(tensor)
}

/// Une inférence (batch 1) : retourne la sortie brute aplatie (rows x 6 : x1, y1, x2, y2, score, classe).
fn run_session(session: &mut Session, input: Vec<f32>) -> anyhow::Result<Vec<f32>> {
    let input_tensor = ort::value::Value::from_array(([1, 3, IMAGE_SIZE, IMAGE_SIZE], input))?;
    let outputs = session.run(ort::inputs!["images" => input_tensor])?;
    let output_tensor = outputs[0].try_extract_tensor::<f32>()?;
    Ok(output_tensor.1.to_vec())
}

//--------------------------------------------------------------------------------------------------

impl TiledYOLO26 {
    pub fn new() -> Self {
        Self::with_config(TiledConfig::default())
    }

    pub fn with_config(cfg: TiledConfig) -> Self {
        TiledYOLO26 {
            session: None,
            classes: Vec::new(),
            cfg,
            rows_per_tile: 0,
            plan: Mutex::new(Vec::new()),
            source_image: Mutex::new(None),
        }
    }

    /// Noms de classes du modèle (utile pour choisir `merge.class_groups`).
    pub fn classes(&self) -> &[String] {
        &self.classes
    }
}

impl ModelPipeline for TiledYOLO26 {
    type Output = Vec<BigDetection>;

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

    /// Ne construit PAS de tenseur : calcule le plan de tuilage et garde l'image source.
    /// Les tenseurs sont créés tuile par tuile dans `infer` (mémoire constante, quelle que soit la taille
    /// de l'image). Le `Vec<f32>` retourné est donc vide et ignoré par `infer`.
    fn preprocess(&self, image: &Mat) -> anyhow::Result<Vec<f32>> {
        let plan = build_plan(
            image.cols(), image.rows(),
            &self.cfg.scales, IMAGE_SIZE, self.cfg.overlap, self.cfg.center_pad,
        );
        if plan.is_empty() {
            return Err(anyhow::anyhow!("Plan de tuilage vide (liste d'échelles vide ?)"));
        }
        *self.plan.lock().unwrap() = plan;
        *self.source_image.lock().unwrap() = Some(image.try_clone()?);

        Ok(Vec::new())
    }

    /// Inférence sur toutes les tuiles. Retourne la concaténation des sorties brutes, dans l'ordre du plan.
    fn infer(
        &mut self,
        _input: &[f32],
    ) -> anyhow::Result<Vec<f32>> {
        let plan: Vec<TilePlan> = self.plan.lock().unwrap().clone();
        if plan.is_empty() {
            return Err(anyhow::anyhow!("Plan de tuilage absent, preprocess doit être appelé avant infer"));
        }
        let scales = self.cfg.scales.clone();

        let session = self
            .session
            .as_mut()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Modèle non chargé, appelle setup_model d'abord"
                )
            })?;

        let guard = self.source_image.lock().unwrap();
        let source: &Mat = (*guard)
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("source_image absent, preprocess doit être appelé avant infer"))?;

        let mut raw: Vec<f32> = Vec::new();
        let mut per_tile: Option<usize> = None;

        for (scale_idx, &scale) in scales.iter().enumerate() {
            // image redimensionnée une seule fois par échelle (échelle 1.0 : on utilise la source telle quelle)
            let resized: Mat;
            let img: &Mat = if (scale - 1.0).abs() < 1e-6 {
                source
            } else {
                let (sw, sh) = scaled_size(source.cols(), source.rows(), scale);
                let mut m = Mat::default();
                imgproc::resize(source, &mut m, Size::new(sw, sh), 0.0, 0.0, imgproc::INTER_AREA)?;
                resized = m;
                &resized
            };

            for t in plan.iter().filter(|t| t.scale_idx == scale_idx) {
                let tile = make_tile(img, t)?;
                let tensor = mat_to_tensor(&tile)?;
                let out = run_session(session, tensor)?;
                println!("Output length : {}", out.len() / 6);

                match per_tile {
                    None => per_tile = Some(out.len()),
                    Some(n) if n != out.len() => {
                        return Err(anyhow::anyhow!(
                            "Taille de sortie incohérente entre tuiles : {} puis {}", n, out.len()
                        ));
                    }
                    _ => {}
                }
                raw.extend_from_slice(&out);
            }
        }

        let n = per_tile.unwrap_or(0);
        if n == 0 || n % 6 != 0 {
            return Err(anyhow::anyhow!(
                "Sortie YOLO26 invalide : {} valeurs par tuile, attendu un multiple de 6", n
            ));
        }
        self.rows_per_tile = n / 6;

        Ok(raw)
    }

    fn postprocess(&self, raw_output: &[f32]) -> anyhow::Result<Self::Output> {
        let rows = self.rows_per_tile;
        let plan: Vec<TilePlan> = self.plan.lock().unwrap().clone();

        if rows == 0 || raw_output.len() != plan.len() * rows * 6 {
            return Err(anyhow::anyhow!(
                "Sortie YOLO26 incohérente : {} valeurs pour {} tuiles x {} lignes x 6",
                raw_output.len(), plan.len(), rows
            ));
        }

        // .take() plutôt que .clone() : Mat n'implémente pas Clone (seulement try_clone)
        let source_image = self.source_image.lock().unwrap().take()
            .ok_or_else(|| anyhow::anyhow!("source_image absent, preprocess doit être appelé avant postprocess"))?;
        let (orig_w, orig_h) = (source_image.cols() as f32, source_image.rows() as f32);

        // 1. lignes brutes -> candidats dans le repère de l'image originale (+ flags de couture)
        let mut cands = Vec::new();
        for (t, tile_rows) in plan.iter().zip(raw_output.chunks_exact(rows * 6)) {
            rows_to_cands(
                tile_rows, t,
                self.cfg.conf, self.cfg.border_margin, self.cfg.drop_border_boxes,
                &mut cands,
            );
        }
        println!("Nombre de candidats avant nettoyage : {}", cands.len());

        // 2. nettoyage : fragments de coutures, boîtes imbriquées, NMS final
        let cands = self.cfg.merge.clean(cands);
        println!("Nombre de candidats après nettoyage : {}", cands.len());

        // --- Conversion vers le type métier utilisé dans le reste du pipeline (identique à YOLO26) ---
        let img_shape = (source_image.rows() as usize, source_image.cols() as usize);

        let detections: Vec<BigDetection> = cands
            .into_iter()
            .filter_map(|c| {
                // clamp pour rester dans les limites de l'image d'origine
                let (x1, y1) = (c.x1.clamp(0.0, orig_w), c.y1.clamp(0.0, orig_h));
                let (x2, y2) = (c.x2.clamp(0.0, orig_w), c.y2.clamp(0.0, orig_h));

                let class_label = self.classes.get(c.cls).cloned()
                    .unwrap_or_else(|| format!("class_{}", c.cls));

                // Si le label ne correspond à aucune DetectionClass connue, on ignore
                // silencieusement cette détection plutôt que de faire planter tout le batch.
                let categorie = DetectionClass::from_str(&class_label).ok()?;

                let bbox = BBox::new(x1, y1, x2, y2, img_shape);

                let cropped = crop_mat(
                    &source_image,
                    (x1 as i32, y1 as i32, x2 as i32, y2 as i32),
                ).ok()?;

                let mut mask = Mask::new();
                mask.setup_mat(cropped);

                // base_name reste vide ici : cette info (Store_point/Shelf_number) est
                // calculée plus haut dans le pipeline Python, pas disponible à ce niveau.
                let mut detection = BigDetection::new(categorie, bbox, mask, String::new());
                detection.set_prediction(c.conf, class_label);

                Some(detection)
            })
            .collect();

        Ok(detections)
    }

    fn unload(&mut self) {
        self.session = None;
    }
}