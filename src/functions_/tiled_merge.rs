//--------------------------------------------------------------------------------------------------
// Filename: tiled_merge.rs
// Rôle    : nettoyage des détections issues d'un tuilage (port de TiledYOLO._merge en Python).
//           Module pur (aucune dépendance à opencv / ort) -> testable seul.
//
// Etapes (dans l'ordre) :
//   1. suppression des fragments (boîtes coupées par une couture de tuile) contenus dans une plus grande
//   2. fusion des fragments alignés qui se font face de part et d'autre d'une couture
//   3. suppression des boîtes imbriquées (morceaux d'un objet plus grand que la tuile, doublons)
//   4. NMS final entre tuiles
//--------------------------------------------------------------------------------------------------

pub const CUT_L: usize = 0;
pub const CUT_T: usize = 1;
pub const CUT_R: usize = 2;
pub const CUT_B: usize = 3;

/// Une détection candidate, dans le repère de l'image ORIGINALE.
#[derive(Debug, Clone, Copy)]
pub struct Cand {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub conf: f32,
    pub cls: usize,
    /// [gauche, haut, droite, bas] : le bord de la boîte colle à une couture de tuile
    /// (bord interne de tuile, pas le bord de l'image).
    pub cut: [bool; 4],
    /// Identifiant de groupe de classes (rempli par `MergeConfig::clean`).
    grp: i64,
}

impl Cand {
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32, conf: f32, cls: usize, cut: [bool; 4]) -> Self {
        Cand { x1, y1, x2, y2, conf, cls, cut, grp: cls as i64 }
    }

    fn area(&self) -> f32 {
        (self.x2 - self.x1).max(0.0) * (self.y2 - self.y1).max(0.0)
    }

    fn any_cut(&self) -> bool {
        self.cut.iter().any(|&c| c)
    }
}

#[derive(Debug, Clone)]
pub struct MergeConfig {
    pub merge_iou: f32,            // NMS final entre tuiles
    pub agnostic: bool,            // NMS / nettoyage indépendants de la classe
    pub merge_fragments: bool,     // active suppression + fusion des fragments aux coutures
    pub contain_thr: f32,          // un fragment contenu à >= 70 % dans une boîte plus grande est supprimé
    pub merge_gap: f32,            // écart max (px) entre deux fragments à fusionner (négatif = recouvrement)
    pub merge_align: f32,          // IoU 1D minimale sur l'axe perpendiculaire à la couture
    pub nested_thr: Option<f32>,   // boîte contenue à >= 85 % dans une boîte >= nested_area_ratio fois plus grande
    pub nested_area_ratio: f32,
    pub class_groups: Vec<Vec<usize>>, // classes que le modèle confond (traitées comme une seule)
}

impl Default for MergeConfig {
    fn default() -> Self {
        MergeConfig {
            merge_iou: 0.5,
            agnostic: false,
            merge_fragments: true,
            contain_thr: 0.7,
            merge_gap: 12.0,
            merge_align: 0.7,
            nested_thr: Some(0.85),
            nested_area_ratio: 2.0,
            class_groups: Vec::new(),
        }
    }
}

fn inter(a: &Cand, b: &Cand) -> f32 {
    let w = (a.x2.min(b.x2) - a.x1.max(b.x1)).max(0.0);
    let h = (a.y2.min(b.y2) - a.y1.max(b.y1)).max(0.0);
    w * h
}

fn iou_1d(lo_a: f32, hi_a: f32, lo_b: f32, hi_b: f32) -> f32 {
    let ov = hi_a.min(hi_b) - lo_a.max(lo_b);
    let un = hi_a.max(hi_b) - lo_a.min(lo_b);
    ov / (un + 1e-9)
}

impl MergeConfig {
    /// Identifiant de groupe : les classes d'un même groupe partagent un id négatif.
    fn group(&self, cls: usize) -> i64 {
        for (gi, g) in self.class_groups.iter().enumerate() {
            if g.contains(&cls) {
                return -(gi as i64 + 1);
            }
        }
        cls as i64
    }

    fn same(&self, a: &Cand, b: &Cand) -> bool {
        self.agnostic || a.grp == b.grp
    }

    /// Point d'entrée : applique les 4 étapes.
    pub fn clean(&self, mut cands: Vec<Cand>) -> Vec<Cand> {
        for c in cands.iter_mut() {
            c.grp = self.group(c.cls);
        }
        if self.merge_fragments {
            cands = self.drop_contained_fragments(cands);
            cands = self.merge_fragment_pairs(cands);
        }
        cands = self.drop_nested(cands);
        self.final_nms(cands)
    }

    /// 1. Supprime une boîte coupée par une couture si elle est contenue dans une boîte plus grande.
    fn drop_contained_fragments(&self, c: Vec<Cand>) -> Vec<Cand> {
        let areas: Vec<f32> = c.iter().map(|x| x.area()).collect();
        let mut keep = vec![true; c.len()];
        for i in 0..c.len() {
            if !c[i].any_cut() {
                continue;
            }
            for j in 0..c.len() {
                if i == j || areas[j] <= areas[i] || !self.same(&c[i], &c[j]) {
                    continue;
                }
                if inter(&c[i], &c[j]) / (areas[i] + 1e-9) >= self.contain_thr {
                    keep[i] = false;
                    break;
                }
            }
        }
        c.into_iter().zip(keep).filter_map(|(x, k)| k.then_some(x)).collect()
    }

    /// 2. Fusionne les paires de fragments alignés qui se font face de part et d'autre d'une couture.
    fn merge_fragment_pairs(&self, mut c: Vec<Cand>) -> Vec<Cand> {
        loop {
            let idx: Vec<usize> = (0..c.len()).filter(|&i| c[i].any_cut()).collect();
            if idx.len() < 2 {
                break;
            }

            // meilleure paire : (score, a, b, couture_verticale)
            let mut best: Option<(f32, usize, usize, bool)> = None;
            for &a in &idx {
                for &b in &idx {
                    if a == b || !self.same(&c[a], &c[b]) {
                        continue;
                    }
                    let (ca, cb) = (&c[a], &c[b]);

                    // a = pièce gauche, b = pièce droite (couture verticale) -> alignement en y
                    if ca.cut[CUT_R] && cb.cut[CUT_L] && cb.x1 - ca.x2 <= self.merge_gap && ca.x1 <= cb.x1 {
                        let s = iou_1d(ca.y1, ca.y2, cb.y1, cb.y2);
                        if s >= self.merge_align && best.map_or(true, |(bs, ..)| s > bs) {
                            best = Some((s, a, b, true));
                        }
                    }
                    // a = pièce haute, b = pièce basse (couture horizontale) -> alignement en x
                    if ca.cut[CUT_B] && cb.cut[CUT_T] && cb.y1 - ca.y2 <= self.merge_gap && ca.y1 <= cb.y1 {
                        let s = iou_1d(ca.x1, ca.x2, cb.x1, cb.x2);
                        if s >= self.merge_align && best.map_or(true, |(bs, ..)| s > bs) {
                            best = Some((s, a, b, false));
                        }
                    }
                }
            }

            let Some((_, a, b, horiz)) = best else { break };
            let (ca, cb) = (c[a], c[b]);
            let winner = if ca.conf >= cb.conf { ca } else { cb };

            let mut cut = [
                ca.cut[0] || cb.cut[0],
                ca.cut[1] || cb.cut[1],
                ca.cut[2] || cb.cut[2],
                ca.cut[3] || cb.cut[3],
            ];
            if horiz {
                // la couture verticale entre les deux pièces est résolue
                cut[CUT_L] = ca.cut[CUT_L];
                cut[CUT_R] = cb.cut[CUT_R];
            } else {
                cut[CUT_T] = ca.cut[CUT_T];
                cut[CUT_B] = cb.cut[CUT_B];
            }

            let merged = Cand {
                x1: ca.x1.min(cb.x1),
                y1: ca.y1.min(cb.y1),
                x2: ca.x2.max(cb.x2),
                y2: ca.y2.max(cb.y2),
                conf: ca.conf.max(cb.conf),
                cls: winner.cls,
                cut,
                grp: winner.grp,
            };

            let (hi, lo) = if a > b { (a, b) } else { (b, a) };
            c.swap_remove(hi);
            c.swap_remove(lo);
            c.push(merged);
        }
        c
    }

    /// 3. Supprime les boîtes quasi entièrement contenues dans une boîte nettement plus grande.
    fn drop_nested(&self, c: Vec<Cand>) -> Vec<Cand> {
        let Some(thr) = self.nested_thr else { return c };
        if c.len() < 2 {
            return c;
        }
        let ratio = self.nested_area_ratio.max(1.01);
        let areas: Vec<f32> = c.iter().map(|x| x.area()).collect();
        let mut keep = vec![true; c.len()];
        for i in 0..c.len() {
            for j in 0..c.len() {
                if i == j || !self.same(&c[i], &c[j]) || areas[j] < ratio * areas[i] {
                    continue;
                }
                if inter(&c[i], &c[j]) / (areas[i] + 1e-9) >= thr {
                    keep[i] = false;
                    break;
                }
            }
        }
        c.into_iter().zip(keep).filter_map(|(x, k)| k.then_some(x)).collect()
    }

    /// 4. NMS final : les boîtes complètes passent avant les boîtes coupées.
    fn final_nms(&self, c: Vec<Cand>) -> Vec<Cand> {
        let areas: Vec<f32> = c.iter().map(|x| x.area()).collect();
        let key = |x: &Cand| x.conf * if x.any_cut() { 0.5 } else { 1.0 };

        let mut order: Vec<usize> = (0..c.len()).collect();
        order.sort_by(|&i, &j| key(&c[j]).partial_cmp(&key(&c[i])).unwrap_or(std::cmp::Ordering::Equal));

        let mut keep: Vec<usize> = Vec::new();
        for &i in &order {
            let cut_i = c[i].any_cut();
            let suppressed = keep.iter().any(|&k| {
                if !self.same(&c[i], &c[k]) {
                    return false;
                }
                let it = inter(&c[i], &c[k]);
                let iou = it / (areas[i] + areas[k] - it + 1e-9);
                let iomin = it / (areas[i].min(areas[k]) + 1e-9);
                iou > self.merge_iou || (cut_i && iomin > 0.8)
            });
            if !suppressed {
                keep.push(i);
            }
        }
        keep.into_iter().map(|i| c[i]).collect()
    }
}

//--------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const NONE: [bool; 4] = [false; 4];
    const R: [bool; 4] = [false, false, true, false];
    const L: [bool; 4] = [true, false, false, false];
    const B: [bool; 4] = [false, false, false, true];
    const T: [bool; 4] = [false, true, false, false];

    fn boxes(v: &[Cand]) -> Vec<[i32; 4]> {
        let mut out: Vec<[i32; 4]> = v.iter().map(|c| [c.x1 as i32, c.y1 as i32, c.x2 as i32, c.y2 as i32]).collect();
        out.sort();
        out
    }

    #[test]
    fn fragment_removed_when_full_box_exists() {
        let cfg = MergeConfig::default();
        let out = cfg.clean(vec![
            Cand::new(600., 100., 640., 500., 0.8, 0, R),
            Cand::new(600., 100., 700., 500., 0.9, 0, NONE),
        ]);
        assert_eq!(boxes(&out), vec![[600, 100, 700, 500]]);
    }

    #[test]
    fn vertical_seam_fragments_are_merged() {
        let cfg = MergeConfig::default();
        let out = cfg.clean(vec![
            Cand::new(300., 600., 640., 900., 0.7, 1, R),
            Cand::new(512., 600., 900., 900., 0.6, 1, L),
        ]);
        assert_eq!(boxes(&out), vec![[300, 600, 900, 900]]);
        assert!((out[0].conf - 0.7).abs() < 1e-6);
    }

    #[test]
    fn horizontal_seam_fragments_are_merged() {
        let cfg = MergeConfig::default();
        let out = cfg.clean(vec![
            Cand::new(1000., 100., 1100., 640., 0.7, 0, B),
            Cand::new(1000., 512., 1100., 800., 0.6, 0, T),
        ]);
        assert_eq!(boxes(&out), vec![[1000, 100, 1100, 800]]);
    }

    #[test]
    fn neighbouring_bottles_are_kept() {
        let cfg = MergeConfig::default();
        let out = cfg.clean(vec![
            Cand::new(1200., 100., 1260., 500., 0.9, 0, NONE),
            Cand::new(1262., 100., 1322., 500., 0.9, 0, NONE),
        ]);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn nested_boxes_are_removed() {
        let cfg = MergeConfig::default();
        let out = cfg.clean(vec![
            Cand::new(0., 100., 150., 2000., 0.7, 0, NONE),   // gros objet
            Cand::new(0., 113., 150., 355., 0.8, 0, NONE),    // morceaux
            Cand::new(0., 505., 148., 685., 0.8, 0, NONE),
            Cand::new(440., 845., 680., 1060., 0.9, 1, NONE), // affiche
            Cand::new(440., 885., 530., 1080., 0.6, 1, NONE), // imbriquée
            Cand::new(800., 100., 900., 400., 0.9, 0, NONE),
        ]);
        assert_eq!(boxes(&out), vec![[0, 100, 150, 2000], [440, 845, 680, 1060], [800, 100, 900, 400]]);
    }

    #[test]
    fn class_groups_remove_cross_class_duplicates() {
        let det = || {
            vec![
                Cand::new(625., 605., 1095., 990., 0.85, 1, NONE),
                Cand::new(740., 610., 1085., 985., 0.60, 3, L),
                Cand::new(1085., 990., 1230., 1300., 0.90, 1, NONE),
                Cand::new(1085., 1000., 1230., 1300., 0.50, 2, NONE),
                Cand::new(1520., 0., 1625., 360., 0.80, 0, NONE),
            ]
        };
        assert_eq!(MergeConfig::default().clean(det()).len(), 5);

        let cfg = MergeConfig { class_groups: vec![vec![1, 2, 3]], ..Default::default() };
        let out = cfg.clean(det());
        assert_eq!(out.len(), 3);
        assert!(out.iter().any(|c| c.cls == 0)); // la classe 0 n'est pas touchée
        assert!(out.iter().all(|c| c.cls != 2 && c.cls != 3)); // la classe d'origine du gagnant est conservée
    }

    #[test]
    fn empty_input() {
        assert!(MergeConfig::default().clean(vec![]).is_empty());
    }
}