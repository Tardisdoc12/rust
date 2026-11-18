//--------------------------------------------------------------------------------------------------
// Filename: identifier.rs
// Author: Jean Anquetil
// Date: Wednesday 18/11/2026
//--------------------------------------------------------------------------------------------------

use opencv::{core::Mat, imgcodecs};

use crate::models::{
    vlm::Candidate as VlmCandidate,
    db::Db,
    embedders::Embedder,
    model_core::ModelPipeline
};

//--------------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Candidate { pub ean: String, pub name: String, pub sim: f32 }

//--------------------------------------------------------------------------------------------------

pub struct Identifier {
    embedder: Embedder,
    db: Db,
    pub k: usize,
}

//--------------------------------------------------------------------------------------------------

impl Identifier {
    /// Candidats prêts pour le VLM : top-k par embedding, avec leur nom et `n_refs` photos de référence chargées.
    pub fn vlm_candidates(&mut self, image: &Mat, n_refs: usize) -> anyhow::Result<Vec<VlmCandidate>> {
        let x = self.preprocess(image)?;
        let q = self.embedder.infer(&x)?;
        let q = self.embedder.postprocess(&q)?;
        let (sims, order, _best) = self.db.rank(&q);
        let mut out = Vec::new();
        for &c in order.iter().take(self.k) {
            let mut refs = Vec::new();
            for p in self.db.pick_refs(&sims, c, n_refs, 0.97) {
                let m = imgcodecs::imread(p, imgcodecs::IMREAD_COLOR)?;   // BGR ; PNG transparents : voir note alpha
                if !m.empty() { refs.push(m); }
            }
            out.push(VlmCandidate { ean: self.db.eans[c].clone(), name: Some(self.db.name(c).to_string()).filter(|s| !s.is_empty()), refs });
        }
        Ok(out)
    }

    pub fn new(db_path: &str, k: usize) -> anyhow::Result<Self> {
        Ok(Self { embedder: Embedder::new(), db: Db::load(db_path)?, k })
    }
    /// Pratique : image -> k candidats.
    pub fn identify(&mut self, image: &Mat) -> anyhow::Result<Vec<Candidate>> {
        let x = self.preprocess(image)?;
        let raw = self.infer(&x)?;
        self.postprocess(&raw)
    }
}

//--------------------------------------------------------------------------------------------------

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