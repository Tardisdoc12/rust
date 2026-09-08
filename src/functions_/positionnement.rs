//--------------------------------------------------------------------------------------------------
// Filename: positionnements.rs
// Author: Jean Anquetil
// Date: 2026-09-08
//--------------------------------------------------------------------------------------------------&

use crate::detections::detections::Detection;
use crate::detections::detection_class::DetectionClass;

//--------------------------------------------------------------------------------------------------

pub fn set_position_on_shelf(detections: &mut Vec<Detection>) {
    detections.sort_by(|a, b| {
        a.shelf_position
            .cmp(&b.shelf_position)
            .then_with(|| a.bbox.xyxyn().0.total_cmp(&b.bbox.xyxyn().0))
    });
    
    let mut current_shelf = None;
    let mut position = 1;

    for det in detections.iter_mut() {
        if current_shelf != Some(det.shelf_position) {
            current_shelf = Some(det.shelf_position);
            position = 1;
        }

        if det.categorie == DetectionClass::Produit || det.categorie == DetectionClass::NoProduct {
            det.position = position;
            position += 1;
        }
    }
}

//--------------------------------------------------------------------------------------------------
/// Clustering 1D équivalent à hcluster.fclusterdata(..., criterion="distance")
/// pour des données 1D : trie puis coupe quand l'écart dépasse le seuil.
fn cluster_indices_1d(values: &[f32], thresh: f32) -> Vec<Vec<usize>> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap());

    let mut clusters: Vec<Vec<usize>> = Vec::new();
    let mut current = vec![order[0]];
    for w in order.windows(2) {
        let (prev, curr) = (w[0], w[1]);
        if (values[curr] - values[prev]).abs() <= thresh {
            current.push(curr);
        } else {
            clusters.push(std::mem::take(&mut current));
            current.push(curr);
        }
    }
    clusters.push(current);
    clusters
}

/// Calcule les bornes de rangées à partir des Y des étiquettes uniquement.
/// Équivalent de `row_clusters` : représentant = Y au plus petit index original
/// du cluster, +0 inséré au début, trié croissant.
pub fn compute_shelf_boundaries(label_ys: &[f32], thresh: f32) -> Vec<f32> {
    if label_ys.len() < 2 {
        // 0 ou 1 étiquette : une seule rangée possible, borne unique à 0
        return vec![0.0];
    }
    let clusters = cluster_indices_1d(label_ys, thresh);
    let mut boundaries: Vec<f32> = clusters
        .iter()
        .map(|idxs| label_ys[*idxs.iter().min().unwrap()])
        .collect();
    boundaries.push(0.0);
    boundaries.sort_by(|a, b| a.partial_cmp(b).unwrap());
    boundaries
}

/// Équivalent de np.digitize(y, boundaries) avec bornes croissantes :
/// renvoie le nombre de bornes <= y, ce qui donne directement le
/// numéro de rangée en base 1 (rangée 0 = au-dessus de la 1ère étiquette).
pub fn shelf_position_for(y: f32, boundaries: &[f32]) -> usize {
    boundaries.iter().filter(|&&b| b <= y).count()
}

/// Assigne shelf_position à toutes les détections qui ne sont pas des étiquettes,
/// en se basant sur le clustering des Y des étiquettes.
/// `is_label` : closure qui identifie une étiquette parmi vos détections.
pub fn assign_shelf_positions(detections: &mut [Detection], thresh: f32) {
    let label_ys: Vec<f32> = detections
        .iter()
        .filter(|d| d.categorie == DetectionClass::Etiquette)
        .map(|d| d.bbox.center_rel().1)
        .collect();

    let boundaries = compute_shelf_boundaries(&label_ys, thresh);

    for det in detections.iter_mut() {
        if det.categorie != DetectionClass::Etiquette && det.categorie != DetectionClass::Reference && det.categorie != DetectionClass::Publicity {
            det.shelf_position = shelf_position_for(det.bbox.center_rel().1, &boundaries) as i32;
        }
    }
}
//--------------------------------------------------------------------------------------------------

pub fn set_all_position(detections: &mut Vec<Detection>) {
    assign_shelf_positions(detections, 0.08 as f32);
    set_position_on_shelf(detections);
}

//--------------------------------------------------------------------------------------------------
// End of file
//--------------------------------------------------------------------------------------------------