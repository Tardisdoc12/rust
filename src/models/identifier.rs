//--------------------------------------------------------------------------------------------------
// Filename: identifier.rs
// Author: Jean Anquetil
// Date: Wednesday 18/11/2026
//--------------------------------------------------------------------------------------------------

use opencv::{core::{self, Mat, Scalar, CV_32F, CV_8U}, imgcodecs, imgproc};
use opencv::prelude::*;

use crate::models::{
    vlm::Candidate as VlmCandidate,
    database::Db,
    embedders::Embedder,
    model_core::ModelPipeline
};


#[derive(Debug, Clone)]
pub struct Candidate { pub ean: String, pub name: String, pub sim: f32 }

/// Seuils de décision (cosinus SigLIP). Valeurs par défaut = AUCUN raccourci : tout part au VLM tant que tu n'as pas
/// calibré (voir la procédure donnée avec ce fichier).
#[derive(Debug, Clone)]
pub struct Gate {
    pub accept_sim: f32,        // top-1 >= ce seuil ET ...
    pub accept_margin: f32,     // ... écart avec le 2e EAN >= ce seuil  => accepté sans VLM
    pub reject_sim: f32,        // top-1 < ce seuil => produit absent de la base, sans VLM
    pub cand_margin: Option<f32>, // n'envoie au VLM que les candidats à moins de ce delta du meilleur
}
impl Default for Gate {
    fn default() -> Self { Self { accept_sim: f32::INFINITY, accept_margin: f32::INFINITY, reject_sim: f32::NEG_INFINITY, cand_margin: None } }
}

/// Règle d'acceptation de la réponse du VLM = celle du pipeline Python (identify_item) :
///   - on suit le VLM seulement s'il est confiant (>= min_conf) ET cohérent entre les ordres ;
///   - sinon on FUSIONNE son avis avec les embeddings : score(c) = P_vlm(c) * exp(fuse_w * (sim(c) - sim_top1) / fuse_t)
///     ("inconnu" : P_vlm(OOB) * exp(fuse_w * (oob_sim - sim_top1) / fuse_t), ou none_prior si oob_sim = None).
/// Défauts = ceux de la commande Python qui fonctionnait (--oob-sim 0.8, none_prior 1.0).
#[derive(Debug, Clone)]
pub struct Decision {
    pub min_conf: f32,
    pub require_agree: bool,
    pub fuse_w: f32,            // 0 = pas de fusion : on garde l'embedding top-1 quand le VLM hésite
    pub fuse_t: f32,
    pub oob_sim: Option<f32>,
    pub none_prior: f32,
}
impl Default for Decision {
    fn default() -> Self {
        Self { min_conf: 0.6, require_agree: true, fuse_w: 1.0, fuse_t: 0.02, oob_sim: Some(0.8), none_prior: 1.0 }
    }
}

pub enum Route {
    Accept { ean: String, name: String, sim: f32, margin: f32 },
    Reject { sim: f32 },
    /// `sims[i]` = similarité embedding de `cands[i]` (même ordre), nécessaire pour la fusion VLM + embedding.
    Ask { cands: Vec<VlmCandidate>, sims: Vec<f32>, sim: f32, margin: f32 },
}

/// Lit une image de référence comme `load_rgb` en Python : les PNG transparents sont composés sur fond BLANC.
/// (IMREAD_COLOR jetait le canal alpha : le fond transparent devenait noir ou du bruit pour le VLM.)
/// IMREAD_UNCHANGED n'applique pas l'orientation EXIF, comme PIL.Image.open.
fn imread_on_white(path: &str) -> anyhow::Result<Mat> {
    let mut m = imgcodecs::imread(path, imgcodecs::IMREAD_UNCHANGED)?;
    if m.empty() {
        return Ok(m);
    }
    if m.depth() != CV_8U {                                   // PNG 16 bits -> 8 bits
        let mut m8 = Mat::default();
        m.convert_to(&mut m8, CV_8U, 1.0 / 256.0, 0.0)?;
        m = m8;
    }
    match m.channels() {
        3 => Ok(m),
        1 => {
            let mut bgr = Mat::default();
            imgproc::cvt_color(&m, &mut bgr, imgproc::COLOR_GRAY2BGR, 0)?;
            Ok(bgr)
        }
        4 => {
            let mut chans = core::Vector::<Mat>::new();
            core::split(&m, &mut chans)?;
            let mut a = Mat::default();
            chans.get(3)?.convert_to(&mut a, CV_32F, 1.0 / 255.0, 0.0)?;
            let mut a3 = Mat::default();
            imgproc::cvt_color(&a, &mut a3, imgproc::COLOR_GRAY2BGR, 0)?;
            let mut bgr = Mat::default();
            imgproc::cvt_color(&m, &mut bgr, imgproc::COLOR_BGRA2BGR, 0)?;
            let mut bgrf = Mat::default();
            bgr.convert_to(&mut bgrf, CV_32F, 1.0, 0.0)?;
            let mut fg = Mat::default();
            core::multiply(&bgrf, &a3, &mut fg, 1.0, -1)?;
            let mut inv = Mat::default();
            core::subtract(&Scalar::all(1.0), &a3, &mut inv, &core::no_array(), -1)?;
            let mut bg = Mat::default();
            inv.convert_to(&mut bg, -1, 255.0, 0.0)?;
            let mut sum = Mat::default();
            core::add(&fg, &bg, &mut sum, &core::no_array(), -1)?;
            let mut out = Mat::default();
            sum.convert_to(&mut out, CV_8U, 1.0, 0.0)?;
            Ok(out)
        }
        n => anyhow::bail!("{path} : nombre de canaux non géré ({n})"),
    }
}

pub struct Identifier {
    embedder: Embedder,
    db: Db,
    pub k: usize,
}

impl Identifier {
    pub fn new(db_path: &str, k: usize) -> anyhow::Result<Self> {
        Ok(Self { embedder: Embedder::new(), db: Db::load(db_path)?, k })
    }

    /// Candidats prêts pour le VLM pour les codes `cs` : nom + `n_refs` photos de référence chargées.
    fn load_cands(&self, sims: &[f32], cs: &[usize], n_refs: usize) -> anyhow::Result<Vec<VlmCandidate>> {
        let mut out = Vec::new();
        for &c in cs {
            let mut refs = Vec::new();
            for p in self.db.pick_refs(sims, c, n_refs, 0.97) {
                let m = imread_on_white(p)?;                    // BGR, fond blanc comme en Python
                if !m.empty() { refs.push(m); }
            }
            out.push(VlmCandidate { ean: self.db.eans[c].clone(), name: Some(self.db.name(c).to_string()).filter(|s| !s.is_empty()), refs });
        }
        Ok(out)
    }

    /// Top-k par embedding -> candidats pour le VLM, sans aucun seuil.
    pub fn vlm_candidates(&mut self, image: &Mat, n_refs: usize) -> anyhow::Result<Vec<VlmCandidate>> {
        let x = self.preprocess(image)?;
        let q = self.embedder.infer(&x)?;
        let q = self.embedder.postprocess(&q)?;
        let (sims, order, _) = self.db.rank(&q);
        let cs: Vec<usize> = order.into_iter().take(self.k).collect();
        self.load_cands(&sims, &cs, n_refs)
    }

    /// Décide s'il faut appeler le VLM. Les photos de référence ne sont chargées que dans le cas `Ask`.
    pub fn route(&mut self, image: &Mat, gate: &Gate, n_refs: usize) -> anyhow::Result<Route> {
        let x = self.preprocess(image)?;
        let q = self.embedder.infer(&x)?;
        let q = self.embedder.postprocess(&q)?;
        let (sims, order, best) = self.db.rank(&q);
        let top = order[0];
        let sim = best[top];
        let margin = order.get(1).map(|&c| sim - best[c]).unwrap_or(f32::INFINITY);   // écart avec le 2e EAN
        if sim < gate.reject_sim {
            return Ok(Route::Reject { sim });
        }
        let name = self.db.name(top).to_string();
        if sim >= gate.accept_sim && margin >= gate.accept_margin {
            return Ok(Route::Accept { ean: self.db.eans[top].clone(), name, sim, margin });
        }
        // on n'envoie au VLM que les candidats proches du meilleur
        let cs: Vec<usize> = order.into_iter().take(self.k)
            .filter(|&c| gate.cand_margin.map_or(true, |m| best[c] >= sim - m)).collect();
        if cs.len() == 1 {                                   // plus qu'un seul candidat plausible : rien à départager
            return Ok(Route::Accept { ean: self.db.eans[top].clone(), name, sim, margin });
        }
        let cand_sims: Vec<f32> = cs.iter().map(|&c| best[c]).collect();
        Ok(Route::Ask { cands: self.load_cands(&sims, &cs, n_refs)?, sims: cand_sims, sim, margin })
    }

    /// Pratique : image -> k candidats.
    pub fn identify(&mut self, image: &Mat) -> anyhow::Result<Vec<Candidate>> {
        let x = self.preprocess(image)?;
        let raw = self.infer(&x)?;
        self.postprocess(&raw)
    }
}

impl ModelPipeline for Identifier {
    type Output = Vec<Candidate>;

    /// model_path = chemin de l'ONNX (la base est chargée dans Identifier::new)
    fn setup_model(&mut self, model_path: &str, device: &str) -> anyhow::Result<()> {
        self.embedder.setup_model(model_path, device)
    }

    // NB : preprocess est &self (trait) alors que ort::Session::run demande &mut self :
    // le calcul GPU ne peut donc pas se faire ici. preprocess ne fait que le tenseur image ;
    // l'embedding q est calculé au début de infer.
    fn preprocess(&self, image: &Mat) -> anyhow::Result<Vec<f32>> {
        self.embedder.preprocess(image)
    }

    /// Une seule image (le tenseur de preprocess). Renvoie la similarité max de chaque EAN (index = code EAN de Db).
    fn infer(&mut self, input: &[f32]) -> anyhow::Result<Vec<f32>> {
        let q = self.embedder.infer(input)?;
        let q = self.embedder.postprocess(&q)?;          // normalisé, 1152 floats
        anyhow::ensure!(q.len() == self.db.dim, "une seule image attendue par appel");
        Ok(self.db.best(&q))
    }

    fn postprocess(&self, raw_output: &[f32]) -> anyhow::Result<Vec<Candidate>> {
        let mut order: Vec<usize> = (0..raw_output.len()).collect();
        order.sort_by(|&a, &b| raw_output[b].partial_cmp(&raw_output[a]).unwrap_or(std::cmp::Ordering::Equal));
        Ok(order.into_iter().take(self.k).map(|c| Candidate {
            ean: self.db.eans[c].clone(), name: self.db.name(c).to_string(), sim: raw_output[c],
        }).collect())
    }

    fn unload(&mut self) { self.embedder.unload() }
}

//--------------------------------------------------------------------------------------------------
// END OF FILE
//--------------------------------------------------------------------------------------------------