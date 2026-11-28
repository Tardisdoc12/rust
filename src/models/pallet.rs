//--------------------------------------------------------------------------------------------------
// Filename: pallet.rs
// Author: Jean Anquetil
// Date: Saturday 28/11/2026
//--------------------------------------------------------------------------------------------------
//! Palette = UN SEUL produit. On ne décide donc pas détection par détection : on pool les preuves de toutes les
//! détections et on prend UNE décision pour la palette entière, appliquée à toutes.
//!
//! Pourquoi pas un vote sur les labels ? « 3 OOB + 1 A » arrive aussi bien quand le produit est INCONNU (A est une
//! erreur, la bonne réponse est OOB) que quand il est connu mais filmé (A est juste, les OOB sont de mauvaises vues).
//! Les labels ne distinguent pas les deux cas ; les similarités, si :
//!   - produit inconnu  : toutes les détections ont une faible similarité avec n'importe quel EAN ;
//!   - produit filmé    : les détections ont une similarité modérée avec le MÊME EAN.
//!
//! Pour chaque EAN e : S(e) = agrégat (moyenne ou médiane) sur les détections de leur meilleure similarité avec e.
//!   S(e*) < reject_sim                          -> OOB pour toute la palette
//!   S(e*) >= accept_sim ET écart avec le 2e >= margin -> e* pour toute la palette
//!   sinon                                       -> ambigu : un seul appel VLM sur la meilleure détection (voir process)
//! Seuils par défaut = None (aucun raccourci, comme `Gate`) : à calibrer sur des palettes annotées.
use serde::Deserialize;

#[derive(Debug, Clone, Copy, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Agg {
    /// moyenne : une détection très bien reconnue (palette filmée avec un produit visible) compte pleinement
    Mean,
    /// médiane : robuste à UNE détection aberrante (un faux A sur une palette inconnue)
    Median,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct PalletParams {
    pub agg: Agg,
    /// S(e*) >= ce seuil (et écart suffisant) => accepté sans VLM. None = jamais de raccourci d'acceptation.
    pub accept_sim: Option<f32>,
    /// S(e*) < ce seuil => palette inconnue (OOB) sans VLM. None = jamais de raccourci de rejet.
    pub reject_sim: Option<f32>,
    /// écart minimal S(e1) - S(e2) pour accepter sans VLM (distingue les variantes d'un même produit)
    pub margin: f32,
    /// cas ambigu : un appel VLM sur la meilleure détection de la palette
    pub use_vlm: bool,
    /// nombre de candidats (EAN) envoyés à ce VLM
    pub vlm_candidates: usize,
}

impl Default for PalletParams {
    fn default() -> Self {
        Self { agg: Agg::Mean, accept_sim: None, reject_sim: None, margin: 0.0, use_vlm: true, vlm_candidates: 4 }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    /// l'EAN `code` (index de la base) pour toute la palette ; `score` = S(code)
    Accept { code: usize, score: f32 },
    /// produit absent de la base ; `score` = S(e*) (la similarité la plus haute de la palette), `code` = cet EAN
    Reject { code: usize, score: f32 },
    /// à arbitrer ; `codes` = candidats triés par S décroissant, `scores[i]` = S(codes[i]) ; `score` = scores[0]
    Ambiguous { codes: Vec<usize>, scores: Vec<f32>, score: f32 },
}

impl Verdict {
    /// EAN le plus proche de la palette (sert à choisir la détection représentante)
    pub fn top_code(&self) -> usize {
        match self {
            Verdict::Accept { code, .. } | Verdict::Reject { code, .. } => *code,
            Verdict::Ambiguous { codes, .. } => codes[0],
        }
    }
}

/// S(e) pour tous les EAN. `bests[d][e]` = meilleure similarité de la détection d avec l'EAN e (même longueur partout).
pub fn pool_scores(bests: &[Vec<f32>], agg: Agg) -> Option<Vec<f32>> {
    let n = bests.first()?.len();
    if n == 0 || bests.iter().any(|b| b.len() != n) {
        return None;
    }
    Some((0..n).map(|e| {
        let mut col: Vec<f32> = bests.iter().map(|b| b[e]).collect();
        match agg {
            Agg::Mean => col.iter().sum::<f32>() / col.len() as f32,
            Agg::Median => median(&mut col),
        }
    }).collect())
}

fn median(v: &mut [f32]) -> f32 {
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

pub fn decide(bests: &[Vec<f32>], p: &PalletParams) -> Option<Verdict> {
    let s = pool_scores(bests, p.agg)?;
    let mut order: Vec<usize> = (0..s.len()).collect();
    order.sort_by(|&a, &b| s[b].total_cmp(&s[a]).then(a.cmp(&b)));      // départage déterministe
    let top = order[0];
    let s1 = s[top];
    let gap = order.get(1).map(|&c| s1 - s[c]).unwrap_or(f32::INFINITY);

    if p.reject_sim.map_or(false, |t| s1 < t) {
        return Some(Verdict::Reject { code: top, score: s1 });
    }
    if p.accept_sim.map_or(false, |t| s1 >= t) && gap >= p.margin {
        return Some(Verdict::Accept { code: top, score: s1 });
    }
    let k = p.vlm_candidates.max(1);
    let codes: Vec<usize> = order.into_iter().take(k).collect();
    let scores: Vec<f32> = codes.iter().map(|&c| s[c]).collect();
    Some(Verdict::Ambiguous { codes, scores, score: s1 })
}

/// Paramètres de la fusion (ceux de `Decision` : fuse_w, fuse_t, oob_sim, none_prior).
#[derive(Debug, Clone, Copy)]
pub struct Fusion {
    pub w: f32,
    pub t: f32,
    pub oob_sim: Option<f32>,
    pub none_prior: f32,
}

/// Fusion VLM x embeddings quand le VLM hésite : score(c) = P_vlm(c) * exp(w * (S(c) - S_top) / t),
/// et pour « inconnu » : P_vlm(OOB) * exp(w * (oob_sim - S_top) / t), ou none_prior si oob_sim = None.
/// `cands` = (EAN, S de la palette). Renvoie (label, probabilité), ou None si le calcul est dégénéré.
pub fn fuse(probs: &std::collections::HashMap<String, f32>, cands: &[(String, f32)], oob_key: &str, f: &Fusion) -> Option<(String, f32)> {
    let s_top = cands.iter().map(|c| c.1).fold(f32::NEG_INFINITY, f32::max);
    let mut fused: Vec<(String, f32)> = cands.iter().map(|(ean, s)| {
        let p = probs.get(ean).copied().unwrap_or(0.0);
        (ean.clone(), p * (f.w * (s - s_top) / f.t).exp())
    }).collect();
    if let Some(&p_oob) = probs.get(oob_key) {
        let prior = match f.oob_sim {
            Some(o) => (f.w * (o - s_top) / f.t).exp(),
            None => f.none_prior,
        };
        fused.push((oob_key.to_string(), p_oob * prior));
    }
    let tot: f32 = fused.iter().map(|x| x.1).sum();
    if !(tot > 0.0 && tot.is_finite()) {
        return None;
    }
    fused.iter().max_by(|a, b| a.1.total_cmp(&b.1)).map(|(k, v)| (k.clone(), v / tot))
}

/// Indice (dans `bests`) de la détection la plus proche de l'EAN `code` : c'est elle qu'on montre au VLM.
pub fn best_crop(bests: &[Vec<f32>], code: usize) -> usize {
    bests.iter().enumerate()
        .max_by(|a, b| a.1[code].total_cmp(&b.1[code]))
        .map(|(i, _)| i).unwrap_or(0)
}

/// Taille réelle de la palette : médiane (largeur, hauteur) des détections qui ont une taille (> 0).
/// (`get_real_size_from_homography` renvoie (0, 0) quand elle n'a pas pu mesurer.)
pub fn median_size(sizes: &[(f32, f32)]) -> Option<(f32, f32)> {
    let mut w: Vec<f32> = sizes.iter().filter(|s| s.0 > 0.0 && s.1 > 0.0).map(|s| s.0).collect();
    let mut h: Vec<f32> = sizes.iter().filter(|s| s.0 > 0.0 && s.1 > 0.0).map(|s| s.1).collect();
    if w.is_empty() { return None; }
    Some((median(&mut w), median(&mut h)))
}

#[cfg(test)]
mod tests {
    use super::*;

    // 3 EAN (A=0, B=1, C=2). Une ligne = une détection : sa meilleure similarité avec A, B, C.
    fn params() -> PalletParams {
        PalletParams { accept_sim: Some(0.60), reject_sim: Some(0.55), margin: 0.05, ..Default::default() }
    }

    #[test]
    fn produit_inconnu_avec_un_faux_a_donne_oob() {
        // chiffres d'illustration : 1 détection à 0.75 avec A (erreur), 3 autres ~0.35
        let b = vec![vec![0.75, 0.30, 0.20], vec![0.35, 0.33, 0.20], vec![0.36, 0.30, 0.22], vec![0.34, 0.31, 0.21]];
        match decide(&b, &params()) {
            Some(Verdict::Reject { code: 0, score }) => assert!((score - 0.45).abs() < 1e-5, "{score}"),
            v => panic!("{v:?}"),
        }
    }

    #[test]
    fn produit_connu_filme_avec_un_seul_visible_donne_a() {
        let b = vec![vec![0.92, 0.30, 0.20], vec![0.60, 0.33, 0.20], vec![0.61, 0.30, 0.22], vec![0.59, 0.31, 0.21]];
        match decide(&b, &params()) {
            Some(Verdict::Accept { code: 0, score }) => assert!((score - 0.68).abs() < 1e-5),
            v => panic!("{v:?}"),
        }
    }

    #[test]
    fn la_mediane_ignore_un_faux_a_isole_la_moyenne_non() {
        let b = vec![vec![0.90, 0.1, 0.1], vec![0.30, 0.1, 0.1], vec![0.30, 0.1, 0.1], vec![0.30, 0.1, 0.1], vec![0.30, 0.1, 0.1]];
        let mean = pool_scores(&b, Agg::Mean).unwrap()[0];
        let med = pool_scores(&b, Agg::Median).unwrap()[0];
        assert!((mean - 0.42).abs() < 1e-5 && (med - 0.30).abs() < 1e-5);
    }

    #[test]
    fn variantes_proches_vont_au_vlm() {
        // A et B presque à égalité : l'écart (0.01) < margin (0.05) => ambigu, même si S est haut
        let b = vec![vec![0.80, 0.79, 0.2], vec![0.78, 0.77, 0.2]];
        match decide(&b, &params()) {
            Some(Verdict::Ambiguous { codes, .. }) => assert_eq!(&codes[..2], &[0, 1]),
            v => panic!("{v:?}"),
        }
    }

    #[test]
    fn zone_grise_entre_les_deux_seuils_va_au_vlm() {
        let b = vec![vec![0.57, 0.2, 0.2], vec![0.57, 0.2, 0.2]];     // 0.55 <= S < 0.60
        assert!(matches!(decide(&b, &params()), Some(Verdict::Ambiguous { .. })));
    }

    #[test]
    fn sans_seuils_tout_part_au_vlm_comme_gate() {
        let b = vec![vec![0.95, 0.1, 0.1], vec![0.95, 0.1, 0.1]];
        assert!(matches!(decide(&b, &PalletParams::default()), Some(Verdict::Ambiguous { .. })));
    }

    #[test]
    fn candidats_limites_et_tries() {
        let b = vec![vec![0.5, 0.9, 0.7], vec![0.5, 0.9, 0.7]];
        match decide(&b, &PalletParams { vlm_candidates: 2, ..Default::default() }) {
            Some(Verdict::Ambiguous { codes, .. }) => assert_eq!(codes, vec![1, 2]),
            v => panic!("{v:?}"),
        }
    }

    #[test]
    fn meilleure_detection_pour_le_vlm() {
        let b = vec![vec![0.60, 0.1, 0.1], vec![0.92, 0.1, 0.1], vec![0.58, 0.1, 0.1]];
        assert_eq!(best_crop(&b, 0), 1);
    }

    #[test]
    fn taille_mediane_ignore_les_mesures_nulles_et_aberrantes() {
        let s = [(10.0, 20.0), (0.0, 0.0), (10.4, 20.2), (30.0, 60.0), (10.2, 19.8)];
        let (w, h) = median_size(&s).unwrap();
        assert!((w - 10.3).abs() < 1e-5 && (h - 20.1).abs() < 1e-5, "{w} {h}");
        assert_eq!(median_size(&[(0.0, 0.0)]), None);
    }

    fn probs(v: &[(&str, f32)]) -> std::collections::HashMap<String, f32> {
        v.iter().map(|(k, p)| (k.to_string(), *p)).collect()
    }
    const FUSION: Fusion = Fusion { w: 1.0, t: 0.02, oob_sim: Some(0.8), none_prior: 1.0 };

    #[test]
    fn fusion_penche_vers_oob_quand_la_palette_est_peu_similaire() {
        // cas du log : S = 0.552 (< oob_sim 0.8), VLM hésitant => on ne force pas un EAN faible
        let cands = vec![("A".to_string(), 0.552), ("B".to_string(), 0.50)];
        let (l, _) = fuse(&probs(&[("A", 0.4), ("B", 0.3), ("OOB", 0.3)]), &cands, "OOB", &FUSION).unwrap();
        assert_eq!(l, "OOB");
    }
    #[test]
    fn fusion_garde_lean_si_similarite_tres_haute() {
        let cands = vec![("A".to_string(), 0.90), ("B".to_string(), 0.60)];
        let (l, p) = fuse(&probs(&[("A", 0.4), ("B", 0.3), ("OOB", 0.3)]), &cands, "OOB", &FUSION).unwrap();
        assert_eq!(l, "A");
        assert!(p > 0.9, "{p}");
    }
    #[test]
    fn fusion_les_embeddings_departagent_un_vlm_indecis() {
        let cands = vec![("A".to_string(), 0.70), ("B".to_string(), 0.60)];
        let f = Fusion { oob_sim: None, ..FUSION };
        let (l, _) = fuse(&probs(&[("A", 0.5), ("B", 0.5)]), &cands, "OOB", &f).unwrap();
        assert_eq!(l, "A");
    }
    #[test]
    fn fusion_degeneree() {
        assert!(fuse(&probs(&[]), &[("A".to_string(), 0.7)], "OOB", &FUSION).is_none());
    }

    #[test]
    fn entrees_incoherentes() {
        assert!(decide(&[], &params()).is_none());
        assert!(decide(&[vec![0.5, 0.5], vec![0.5]], &params()).is_none());
    }

    #[test]
    fn config_json_partielle() {
        let p: PalletParams = serde_json::from_str(r#"{"agg":"median","accept_sim":0.6,"reject_sim":0.5}"#).unwrap();
        assert_eq!(p.agg, Agg::Median);
        assert_eq!(p.accept_sim, Some(0.6));
        assert!(p.use_vlm && p.vlm_candidates == 4);
    }
}

//--------------------------------------------------------------------------------------------------
// END OF FILE
//--------------------------------------------------------------------------------------------------