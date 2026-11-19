//--------------------------------------------------------------------------------------------------
// Filename: vlm.rs
// Author: Jean Anquetil
// Date: Wednesday 18/11/2026
//--------------------------------------------------------------------------------------------------

use std::collections::HashMap;
use opencv::{core::{Mat, Size}, imgproc, prelude::*};

pub const LETTERS: &str = "ABCDEFGHI";
pub const NONE_LETTER: char = 'Z';
pub const OOB: &str = "OOB";

pub const INSTRUCTION: &str = concat!(
    "Which option shows the same exact product as the query image? The options may be near-identical ",
    "variants from the same range (flavour, size, version), so do not rely on the overall look: compare ",
    "the text printed on the packaging (variant name, flavour, weight or volume, claims), the colours of ",
    "bands, caps and labels, and small graphic details. Ignore lighting, background, reflections and ",
    "viewing angle."
);

// ------------------------------------------------------------------ types

pub struct Candidate {
    pub ean: String,
    pub name: Option<String>,
    pub refs: Vec<Mat>,          // photos de référence (BGR, comme imread) ; 1 à 3 suffisent
}

#[derive(Debug, Clone)]
pub struct VlmResult {
    pub ean: Option<String>,     // None = "aucun de ces produits" (OOB)
    pub conf: f32,               // probabilité moyenne du vainqueur (pas une garantie d'exactitude)
    pub probs: HashMap<String, f32>, // par EAN + "OOB"
    pub agree: bool,             // tous les ordres de présentation ont le même vainqueur
    pub n_ok: usize,
}

pub enum Part { Text(String), Image(usize) }   // Image(i) -> prompt.images[i]
pub struct Prompt<'a> { pub parts: Vec<Part>, pub images: Vec<&'a Mat> }

/// Ce que doit fournir un moteur : les logits du PREMIER token généré après le prompt (l'habillage chat template
/// est de son ressort) et les ids de token qui valent pour chaque lettre.
pub trait VlmBackend {
    fn first_token_logits(&mut self, prompt: &Prompt) -> anyhow::Result<Vec<f32>>;
    fn letter_ids(&self) -> &HashMap<char, Vec<u32>>;
}

// ------------------------------------------------------------------ utilitaires

/// Réduit (jamais n'agrandit) pour que le grand côté soit <= max_side, ratio conservé (équiv. thumbnail).
fn fit_max_side(m: &Mat, max_side: i32) -> anyhow::Result<Mat> {
    let (w, h) = (m.cols(), m.rows());
    if w.max(h) <= max_side { return Ok(m.try_clone()?); }
    let k = max_side as f64 / w.max(h) as f64;
    let mut out = Mat::default();
    imgproc::resize(m, &mut out, Size::new(((w as f64 * k).round() as i32).max(1), ((h as f64 * k).round() as i32).max(1)),
                    0.0, 0.0, imgproc::INTER_AREA)?;
    Ok(out)
}

/// Ordres de présentation : l'ordre donné, son inverse, puis des mélanges déterministes (graine).
fn make_orders(n: usize, perms: usize, seed: u64) -> Vec<Vec<usize>> {
    let base: Vec<usize> = (0..n).collect();
    let mut orders = vec![base.clone(), base.iter().rev().cloned().collect()];
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) | 1;
    let mut next = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
    while orders.len() < perms {
        let mut o = base.clone();
        for i in (1..o.len()).rev() { o.swap(i, (next() % (i as u64 + 1)) as usize); }
        orders.push(o);
    }
    orders.truncate(perms.max(1));
    orders
}

/// Softmax sur tout le vocabulaire, somme des probas des ids de chaque lettre, renormalisée sur les lettres valides.
/// None si le modèle ne répond pas par une lettre (masse < min_mass).
fn probs_from_logits(logits: &[f32], ids: &HashMap<char, Vec<u32>>, letters: &[char], min_mass: f64) -> Option<HashMap<char, f64>> {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = m + logits.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln();
    let p: HashMap<char, f64> = letters.iter().map(|l| {
        let s = ids.get(l).map(|v| v.iter().filter_map(|&i| logits.get(i as usize)).map(|&x| (x as f64 - lse).exp()).sum()).unwrap_or(0.0);
        (*l, s)
    }).collect();
    let mass: f64 = p.values().sum();
    if mass < min_mass { return None; }
    Some(p.into_iter().map(|(k, v)| (k, v / mass)).collect())
}

// ------------------------------------------------------------------ la classe

pub struct Vlm {
    backend: Box<dyn VlmBackend + Send>,
    pub perms: usize,
    pub early_conf: Option<f32>,
    pub max_side: i32,
    pub no_names: bool,
    pub seed: u64,
}

impl Vlm {
    pub fn new(backend: Box<dyn VlmBackend + Send>) -> Self {
        Self { backend, perms: 2, early_conf: None, max_side: 672, no_names: false, seed: 0 }
    }

    /// Même texte que `_content` du Python. `order` = indices dans `cands`, dans l'ordre de présentation.
    fn build_prompt<'a>(&self, query: &'a Mat, cands: &[Candidate], refs: &'a [Vec<Mat>], order: &[usize]) -> (Prompt<'a>, Vec<char>) {
        let letters: Vec<char> = LETTERS.chars().take(order.len()).collect();
        let z = NONE_LETTER;
        let mut parts = vec![Part::Text("Query image (a product photographed on a store shelf, automatically cropped by a detector; \
it may be cut off, partly hidden or blurred):".into())];
        let mut images: Vec<&Mat> = vec![query];
        parts.push(Part::Image(0));
        for (&l, &c) in letters.iter().zip(order) {
            let nm = if self.no_names { None } else { cands[c].name.as_deref().filter(|s| !s.is_empty()) };
            parts.push(Part::Text(format!("Option {l}{} - reference photo(s) of this product:",
                nm.map(|n| format!(" ({n})")).unwrap_or_default())));
            for r in &refs[c] { images.push(r); parts.push(Part::Image(images.len() - 1)); }
        }
        parts.push(Part::Text(format!("Option {z}: none of the options above. Choose {z} if the query is a different product, OR if the \
query image does not show enough of the packaging (cut off, hidden, blurred, wrong side) to be \
sure which option it is.")));
        parts.push(Part::Text(format!("{INSTRUCTION} Choose a letter {}-{} only if the visible packaging clearly \
matches that exact product; when in doubt, answer {z}. Answer with the single letter of the matching option.",
            letters[0], letters[letters.len() - 1])));
        let mut all = letters; all.push(z);
        (Prompt { parts, images }, all)
    }

    /// candidats : du meilleur au moins bon (max 9), chacun avec ses photos de référence.
    pub fn process_product(&mut self, query: &Mat, cands: &[Candidate]) -> anyhow::Result<VlmResult> {
        anyhow::ensure!(!cands.is_empty() && cands.len() <= LETTERS.len(), "1 à {} candidats attendus", LETTERS.len());
        if cands.len() == 1 {
            return Ok(VlmResult { ean: Some(cands[0].ean.clone()), conf: 1.0, probs: HashMap::from([(cands[0].ean.clone(), 1.0)]), agree: true, n_ok: 0 });
        }
        let q = fit_max_side(query, self.max_side)?;
        let refs: Vec<Vec<Mat>> = cands.iter().map(|c| c.refs.iter().map(|r| fit_max_side(r, self.max_side)).collect::<anyhow::Result<Vec<_>>>())
            .collect::<anyhow::Result<_>>()?;

        let mut acc: HashMap<String, f32> = HashMap::new();
        let mut winners: Vec<String> = Vec::new();
        for order in make_orders(cands.len(), self.perms, self.seed) {
            let (prompt, letters) = self.build_prompt(&q, cands, &refs, &order);
            let probs = match self.backend.first_token_logits(&prompt)
                .map(|lg| probs_from_logits(&lg, self.backend.letter_ids(), &letters, 0.3)) {
                Ok(Some(p)) => p,
                Ok(None) => continue,
                Err(e) => { eprintln!("[err VLM] {e}"); continue; }
            };
            let mut by: HashMap<String, f32> = HashMap::new();
            for (i, &c) in order.iter().enumerate() { by.insert(cands[c].ean.clone(), probs[&letters[i]] as f32); }
            by.insert(OOB.into(), probs[&NONE_LETTER] as f32);
            for (k, v) in &by { *acc.entry(k.clone()).or_insert(0.0) += v; }
            let (wk, wv) = by.iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(k, v)| (k.clone(), *v)).unwrap();
            winners.push(wk);
            if let Some(t) = self.early_conf { if wv >= t && winners.len() < self.perms { break; } }
        }
        anyhow::ensure!(!winners.is_empty(), "le VLM n'a pas pu répondre");
        let n = winners.len() as f32;
        let avg: HashMap<String, f32> = acc.into_iter().map(|(k, v)| (k, v / n)).collect();
        let (win, conf) = avg.iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(k, v)| (k.clone(), *v)).unwrap();
        Ok(VlmResult {
            ean: if win == OOB { None } else { Some(win) }, conf, agree: winners.iter().all(|w| *w == winners[0]), n_ok: winners.len(),
            probs: avg,
        })
    }
}
//--------------------------------------------------------------------------------------------------
// END OF FILE
//--------------------------------------------------------------------------------------------------