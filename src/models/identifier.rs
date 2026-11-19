//--------------------------------------------------------------------------------------------------
// Filename: identifier.rs
// Author: Jean Anquetil
// Date: Wednesday 18/11/2026
//--------------------------------------------------------------------------------------------------

use opencv::{core::Mat, imgcodecs};
use opencv::prelude::MatTraitConst;

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

pub enum Route {
    Accept { ean: String, name: String, sim: f32, margin: f32 },
    Reject { sim: f32 },
    Ask { cands: Vec<VlmCandidate>, sim: f32, margin: f32 },
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
                let m = imgcodecs::imread(p, imgcodecs::IMREAD_COLOR)?;   // BGR ; PNG transparents : voir note alpha
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
        Ok(Route::Ask { cands: self.load_cands(&sims, &cs, n_refs)?, sim, margin })
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