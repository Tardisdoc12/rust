//--------------------------------------------------------------------------------------------------
// Filename: tiled_geometry.rs
// Rôle    : géométrie du tuilage (port de TiledYOLO._positions / _make_tile / _infer_batch en Python).
//           Module pur (aucune dépendance à opencv / ort) -> testable seul.
//--------------------------------------------------------------------------------------------------

use super::tiled_merge::Cand;

/// Une tuile à inférer : position dans l'image (éventuellement redimensionnée) et letterbox.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TilePlan {
    pub scale_idx: usize,  // index dans la liste des échelles
    pub scale: f32,
    pub x0: i32,           // position de la tuile dans l'image redimensionnée
    pub y0: i32,
    pub w: i32,            // taille réelle du contenu (avant padding)
    pub h: i32,
    pub pad_left: i32,     // padding ajouté dans la tuile
    pub pad_top: i32,
    pub img_w: i32,        // taille de l'image redimensionnée
    pub img_h: i32,
}

/// Identique à TiledYOLO._positions.
/// `overlap` est en f64 (comme en Python) : en f32, 0.3 devient 0.30000001 et le pas passe de 448 à 447 px.
pub fn positions(length: i32, tile: i32, overlap: f64) -> Vec<i32> {
    let stride = ((tile as f64 * (1.0 - overlap)) as i32).max(1);
    let mut pos = vec![0];
    while *pos.last().unwrap() + tile < length {
        pos.push(pos.last().unwrap() + stride);
    }
    pos
}

/// Taille de l'image après redimensionnement (comme cv2.resize avec fx / fy).
pub fn scaled_size(width: i32, height: i32, scale: f32) -> (i32, i32) {
    if (scale - 1.0).abs() < 1e-6 {
        return (width, height);
    }
    (
        ((width as f32 * scale).round() as i32).max(1),
        ((height as f32 * scale).round() as i32).max(1),
    )
}

/// Liste toutes les tuiles, échelle par échelle (ordre = ordre d'inférence).
pub fn build_plan(width: i32, height: i32, scales: &[f32], tile: i32, overlap: f64, center_pad: bool) -> Vec<TilePlan> {
    let mut plan = Vec::new();
    for (scale_idx, &scale) in scales.iter().enumerate() {
        let (sw, sh) = scaled_size(width, height, scale);
        for y0 in positions(sh, tile, overlap) {
            for x0 in positions(sw, tile, overlap) {
                let w = tile.min(sw - x0);
                let h = tile.min(sh - y0);
                let (pad_left, pad_top) = if w == tile && h == tile {
                    (0, 0)
                } else if center_pad {
                    ((tile - w) / 2, (tile - h) / 2)
                } else {
                    (0, 0)
                };
                plan.push(TilePlan { scale_idx, scale, x0, y0, w, h, pad_left, pad_top, img_w: sw, img_h: sh });
            }
        }
    }
    plan
}

/// Convertit les lignes brutes d'une tuile [x1, y1, x2, y2, score, classe] (repère tuile letterboxée)
/// en candidats dans le repère de l'image ORIGINALE, avec les flags de couture.
pub fn rows_to_cands(
    rows: &[f32],
    t: &TilePlan,
    conf_threshold: f32,
    border_margin: f32,
    mark_cut: bool,
    out: &mut Vec<Cand>,
) {
    let (w, h) = (t.w as f32, t.h as f32);
    let m = border_margin;

    for r in rows.chunks_exact(6) {
        let score = r[4];
        if score < conf_threshold {
            continue;
        }

        // repère de la tuile -> repère du contenu (on retire le padding), puis clip au contenu réel
        let x1 = (r[0] - t.pad_left as f32).clamp(0.0, w);
        let y1 = (r[1] - t.pad_top as f32).clamp(0.0, h);
        let x2 = (r[2] - t.pad_left as f32).clamp(0.0, w);
        let y2 = (r[3] - t.pad_top as f32).clamp(0.0, h);

        // boîtes dans le padding ou dégénérées
        if (x2 - x1) <= 1.0 || (y2 - y1) <= 1.0 {
            continue;
        }

        // la boîte colle à une couture (bord interne de tuile, pas le bord de l'image)
        let cut = if mark_cut {
            [
                t.x0 > 0 && x1 < m,
                t.y0 > 0 && y1 < m,
                t.x0 + t.w < t.img_w && x2 > w - m,
                t.y0 + t.h < t.img_h && y2 > h - m,
            ]
        } else {
            [false; 4]
        };

        // repère de l'image redimensionnée -> repère de l'image originale
        let (ox, oy) = (t.x0 as f32, t.y0 as f32);
        let s = t.scale;
        out.push(Cand::new(
            (x1 + ox) / s,
            (y1 + oy) / s,
            (x2 + ox) / s,
            (y2 + oy) / s,
            score,
            r[5] as usize,
            cut,
        ));
    }
}

//--------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positions_match_python() {
        // Python : _positions(1500) avec tile=640, overlap=0.3 -> [0, 448, 896]
        assert_eq!(positions(1500, 640, 0.3), vec![0, 448, 896]);
        assert_eq!(positions(640, 640, 0.3), vec![0]);
        assert_eq!(positions(500, 640, 0.3), vec![0]);
        assert_eq!(positions(641, 640, 0.2), vec![0, 512]);
    }

    #[test]
    fn plan_edge_tiles_are_letterboxed() {
        let plan = build_plan(1000, 700, &[1.0], 640, 0.3, true);
        assert_eq!(plan.len(), 4); // x: [0, 448] ; y: [0, 448]
        let t = plan.iter().find(|t| t.x0 == 448 && t.y0 == 448).unwrap();
        assert_eq!((t.w, t.h), (552, 252));
        assert_eq!((t.pad_left, t.pad_top), (44, 194));
        let full = plan.iter().find(|t| t.x0 == 0 && t.y0 == 0).unwrap();
        assert_eq!((full.w, full.h, full.pad_left, full.pad_top), (640, 640, 0, 0));
    }

    #[test]
    fn plan_multi_scale() {
        let plan = build_plan(1000, 700, &[1.0, 0.5], 640, 0.3, true);
        assert_eq!(plan.len(), 5); // 4 tuiles à l'échelle 1 + 1 tuile (500x350) à l'échelle 0.5
        let t = plan.last().unwrap();
        assert_eq!((t.scale_idx, t.img_w, t.img_h, t.w, t.h), (1, 500, 350, 500, 350));
        assert_eq!((t.pad_left, t.pad_top), (70, 145));
    }

    #[test]
    fn rows_are_mapped_back_with_seam_flags() {
        let plan = build_plan(1500, 700, &[1.0], 640, 0.3, true);
        // tuile du milieu en x : x0 = 448, y0 = 0 (h = 640 -> y0 = 0 couvre 0..640 < 700, donc couture en bas)
        let t = plan.iter().find(|t| t.x0 == 448 && t.y0 == 0).unwrap();
        let mut out = Vec::new();
        // boîte collée au bord gauche de la tuile et au bord bas ; ligne 2 sous le seuil ; ligne 3 dans le padding
        let rows = [
            0.0, 100.0, 80.0, 639.0, 0.9, 2.0, //
            10.0, 10.0, 50.0, 50.0, 0.1, 0.0, //
            10.0, 10.0, 10.5, 50.0, 0.9, 0.0,
        ];
        rows_to_cands(&rows, t, 0.3, 6.0, true, &mut out);
        assert_eq!(out.len(), 1);
        let c = out[0];
        assert_eq!((c.x1, c.y1, c.x2, c.y2, c.cls), (448.0, 100.0, 528.0, 639.0, 2));
        assert_eq!(c.cut, [true, false, false, true]);
    }

    #[test]
    fn rows_scaled_and_image_border_not_a_seam() {
        let plan = build_plan(1000, 700, &[0.5], 640, 0.3, true);
        assert_eq!(plan.len(), 1);
        let t = &plan[0]; // image 500x350 letterboxée, pad (70, 145)
        let mut out = Vec::new();
        // boîte dans le repère tuile : contenu (0..500, 0..350) décalé de (70,145)
        rows_to_cands(&[70.0, 145.0, 270.0, 345.0, 0.8, 1.0], t, 0.3, 6.0, true, &mut out);
        let c = out[0];
        assert_eq!((c.x1, c.y1, c.x2, c.y2), (0.0, 0.0, 400.0, 400.0)); // retour en échelle 1.0
        assert_eq!(c.cut, [false; 4]); // bord de l'image, pas une couture
    }
}