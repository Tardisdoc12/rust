//--------------------------------------------------------------------------------------------------
// Filename: price_ocr.rs
// Port de bbox_to_string.py
//--------------------------------------------------------------------------------------------------

use std::collections::HashSet;

const Y_TOLERANCE: f32 = 15.0;
const GAP_RATIO: f32 = 1.5;
const DEDUP_IOU_THRESHOLD: f32 = 0.4;

fn ignored_classes() -> HashSet<&'static str> {
    HashSet::from(["noise"])
}

fn class_to_char(name: &str) -> String {
    match name {
        // "." et pas "," : `str::parse::<f32>` (safe_float) ne comprend pas la
        // virgule → "1575,00" donnait 0.0.
        "virgule" | "," => ".".to_string(),
        other => other.to_string(),
    }
}

fn is_number(class_name: &str) -> bool {
    class_name.len() == 1 && class_name.chars().next().unwrap().is_ascii_digit()
}

#[derive(Debug, Clone)]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

#[derive(Debug, Clone)]
pub struct Detection {
    pub bbox: BoundingBox,
    pub score: f32,
    pub class_label: String,
}

//--------------------------------------------------------------------------------------------------
// IoU + déduplication
//--------------------------------------------------------------------------------------------------

fn iou(a: &BoundingBox, b: &BoundingBox) -> f32 {
    let ix1 = a.x1.max(b.x1);
    let iy1 = a.y1.max(b.y1);
    let ix2 = a.x2.min(b.x2);
    let iy2 = a.y2.min(b.y2);

    let iw = (ix2 - ix1).max(0.0);
    let ih = (iy2 - iy1).max(0.0);
    let inter = iw * ih;

    if inter <= 0.0 {
        return 0.0;
    }

    let area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    let area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    let union = area_a + area_b - inter;

    if union > 0.0 { inter / union } else { 0.0 }
}

/// Supprime les doublons quasi-identiques, garde celui avec la plus haute confiance.
fn deduplicate_boxes(mut detections: Vec<Detection>) -> Vec<Detection> {
    detections.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    let mut kept: Vec<Detection> = Vec::new();
    for d in detections {
        let overlaps_existing = kept.iter().any(|k| iou(&d.bbox, &k.bbox) >= DEDUP_IOU_THRESHOLD);
        if !overlaps_existing {
            kept.push(d);
        }
    }
    kept
}

//--------------------------------------------------------------------------------------------------
// Filtrage principal
//--------------------------------------------------------------------------------------------------

/// Enlève uniquement le bruit ("noise"), garde chiffres, "-" et "virgule".
pub fn filter_detections(detections: Vec<Detection>) -> Vec<Detection> {
    let ignored = ignored_classes();

    let kept: Vec<Detection> = detections
        .into_iter()
        .filter(|d| !ignored.contains(d.class_label.as_str()))
        .collect();

    let mut kept = deduplicate_boxes(kept);
    kept.sort_by(|a, b| a.bbox.x1.partial_cmp(&b.bbox.x1).unwrap());
    kept
}

fn y_center(d: &Detection) -> f32 {
    (d.bbox.y1 + d.bbox.y2) / 2.0
}

/// Regroupe les détections par ligne horizontale (clustering sur le centre Y),
/// garde la ligne la plus peuplée.
pub fn filter_to_main_y_line(detections: Vec<Detection>, y_tolerance: f32) -> Vec<Detection> {
    if detections.is_empty() {
        return detections;
    }

    let mut sorted = detections;
    sorted.sort_by(|a, b| y_center(a).partial_cmp(&y_center(b)).unwrap());

    let mut clusters: Vec<Vec<Detection>> = vec![vec![sorted[0].clone()]];

    for d in sorted.into_iter().skip(1) {
        let last = clusters.last().unwrap();
        let cluster_mean_y: f32 = last.iter().map(y_center).sum::<f32>() / last.len() as f32;

        // En plus de la tolérance fixe (15 px), on accepte tout élément qui
        // recouvre verticalement la ligne sur au moins la moitié de sa hauteur :
        // des centimes en exposant sont entièrement contenus dans la hauteur des
        // gros chiffres, mais leur centre Y peut être décalé de bien plus de
        // 15 px sur un crop haute résolution (ils étaient alors jetés).
        let span_y1 = last.iter().map(|c| c.bbox.y1).fold(f32::INFINITY, f32::min);
        let span_y2 = last.iter().map(|c| c.bbox.y2).fold(f32::NEG_INFINITY, f32::max);
        let overlap = d.bbox.y2.min(span_y2) - d.bbox.y1.max(span_y1);
        let same_line_by_overlap = overlap >= 0.5 * height(&d).min(span_y2 - span_y1);

        if (y_center(&d) - cluster_mean_y).abs() <= y_tolerance || same_line_by_overlap {
            clusters.last_mut().unwrap().push(d);
        } else {
            clusters.push(vec![d]);
        }
    }

    let mut best_cluster = clusters
        .into_iter()
        .max_by_key(|c| c.len())
        .unwrap_or_default();

    best_cluster.sort_by(|a, b| a.bbox.x1.partial_cmp(&b.bbox.x1).unwrap());
    best_cluster
}

//--------------------------------------------------------------------------------------------------
// Regroupement en "blocs" de chiffres proches (séparés par des espaces significatifs)
//--------------------------------------------------------------------------------------------------

pub fn group_digits(detections: &[Detection], gap_ratio: f32) -> Vec<Vec<Detection>> {
    if detections.is_empty() {
        return Vec::new();
    }

    let widths: Vec<f32> = detections.iter().map(|d| d.bbox.x2 - d.bbox.x1).collect();
    let avg_width: f32 = widths.iter().sum::<f32>() / widths.len() as f32;
    let threshold = gap_ratio * avg_width;

    let mut groups: Vec<Vec<Detection>> = vec![vec![detections[0].clone()]];

    for pair in detections.windows(2) {
        let prev = &pair[0];
        let curr = &pair[1];
        let gap = curr.bbox.x1 - prev.bbox.x2;

        if gap > threshold {
            groups.push(vec![curr.clone()]);
        } else {
            groups.last_mut().unwrap().push(curr.clone());
        }
    }

    groups
}

pub fn pick_largest_group(groups: Vec<Vec<Detection>>) -> Vec<Detection> {
    groups.into_iter().max_by_key(|g| g.len()).unwrap_or_default()
}

//--------------------------------------------------------------------------------------------------
// Reconstruction de la chaîne de prix
//--------------------------------------------------------------------------------------------------

/// Un chiffre est considéré comme "centime en exposant" si sa hauteur est
/// inférieure à ce ratio de la hauteur des chiffres principaux qui précèdent.
const CENTS_HEIGHT_RATIO: f32 = 0.75;

fn is_separator(label: &str) -> bool {
    matches!(label, "." | "," | "virgule")
}

fn height(d: &Detection) -> f32 {
    d.bbox.y2 - d.bbox.y1
}

/// Détecte une séparation décimale implicite (centimes écrits plus petits /
/// en exposant) et insère une "virgule" synthétique.
///
/// Corrige l'ancien port Python :
///  - la référence n'est plus `group[0]` mais la hauteur MAX des chiffres
///    principaux déjà vus (un "1" en tête faussait tout) ;
///  - on ne compare plus la LARGEUR : un "1" est étroit et des "00" en exposant
///    sont souvent fusionnés en une seule boîte large → test inutilisable ;
///  - plus d'indexation négative façon Python (`group[p-1]` avec p == 0) ;
///  - on n'invente rien si un vrai séparateur décimal existe déjà
///    (1 ou 2 chiffres après lui).
pub fn group_to_price_str(mut group: Vec<Detection>) -> String {
    if group.is_empty() {
        return String::new();
    }

    let n = group.len();
    let has_explicit_decimal = group.iter().enumerate().any(|(i, d)| {
        let after = n - (i + 1);
        is_separator(&d.class_label) && (1..=2).contains(&after)
    });

    let mut insert_index: Option<usize> = None;
    if !has_explicit_decimal {
        let mut ref_height = 0.0f32;
        for (i, d) in group.iter().enumerate() {
            if !is_number(&d.class_label) {
                continue;
            }
            let h = height(d);
            if ref_height > 0.0 && h < CENTS_HEIGHT_RATIO * ref_height {
                insert_index = Some(i);
                break;
            }
            ref_height = ref_height.max(h);
        }
    }

    if let Some(idx) = insert_index {
        let prev_box = &group[idx - 1].bbox;
        let curr_box = &group[idx].bbox;

        let mid_x = ((prev_box.x2 + curr_box.x1) / 2.0).floor();
        let mid_y = ((prev_box.y1 + curr_box.y2) / 2.0).floor();

        let virgule = Detection {
            bbox: BoundingBox {
                x1: mid_x,
                y1: mid_y,
                x2: mid_x + 1.0,
                y2: mid_y + 1.0,
            },
            score: 0.90,
            class_label: "virgule".to_string(),
        };

        group.insert(idx, virgule);
    }

    group_to_string(&group)
}

fn group_to_string(group: &[Detection]) -> String {
    let ignored = ignored_classes();
    let mut price = String::new();

    for (i, digit) in group.iter().enumerate() {
        if digit.class_label == "." {
            let remaining_after = group.len() - (i + 1);
            if remaining_after > 2 {
                continue; // probable séparateur de milliers, pas un point décimal
            }
        }

        if ignored.contains(digit.class_label.as_str()) {
            continue;
        }

        price.push_str(&class_to_char(&digit.class_label));
    }

    price
}

//--------------------------------------------------------------------------------------------------
// Pipeline complet, équivalent de _postprocess côté Python
//--------------------------------------------------------------------------------------------------

pub struct PriceResult {
    pub detections: Vec<Detection>,
}

pub fn extract_price(raw_detections: Vec<Detection>) -> PriceResult {
    let detections = filter_detections(raw_detections);
    let detections = filter_to_main_y_line(detections, Y_TOLERANCE);

    let groups = group_digits(&detections, GAP_RATIO);
    let mut best_group = pick_largest_group(groups);

    // enlève les éléments de tête tant qu'ils ne sont pas un chiffre
    while let Some(first) = best_group.first() {
        if is_number(&first.class_label) {
            break;
        }
        best_group.remove(0);
    }

    PriceResult {
        detections: best_group,
    }
}