//--------------------------------------------------------------------------------------------------
// Filename: utils.rs
// Author: Jean Anquetil
// Date: 2026-09-28
//--------------------------------------------------------------------------------------------------

use opencv::core::{Mat, Point2f, Vector};
use opencv::prelude::MatTraitConst;

use crate::detections::mask::Mask;

//--------------------------------------------------------------------------------------------------

pub fn safe_float(x: &str) -> f32 {
    x.parse::<f32>().unwrap_or(0.0)
}

//--------------------------------------------------------------------------------------------------

pub fn compute_homography_from_reference(
    ref_mask: &Mask,
    ref_real_cm: f32,
) -> anyhow::Result<Option<(Mat, f32)>> {
    let corners_opt = ref_mask.get_reference_corners_mat()?;
    let corners = match corners_opt {
        Some(c) if c.len() == 4 => c,
        _ => return Ok(None),
    };

    let src = order_points(&corners);

    let dist = |a: (f32, f32), b: (f32, f32)| ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt();

    let side1 = dist(src[0], src[1]);
    let side2 = dist(src[1], src[2]);
    let side3 = dist(src[2], src[3]);
    let side4 = dist(src[3], src[0]);
    let ref_px = (side1 + side2 + side3 + side4) / 4.0;

    let cm_per_pixel = ref_real_cm / ref_px;

    let vec_x = ((src[1].0 - src[0].0) / side1, (src[1].1 - src[0].1) / side1);
    let vec_y = ((src[3].0 - src[0].0) / side4, (src[3].1 - src[0].1) / side4);

    let h_img = ref_mask.mat.rows() as f32;
    let w_img = ref_mask.mat.cols() as f32;
    let origin = src[0];

    let project_point = |pt: (f32, f32)| -> Point2f {
        let delta = (pt.0 - origin.0, pt.1 - origin.1);
        let x_cm = (delta.0 * vec_x.0 + delta.1 * vec_x.1) * cm_per_pixel;
        let y_cm = (delta.0 * vec_y.0 + delta.1 * vec_y.1) * cm_per_pixel;
        Point2f::new(x_cm, y_cm)
    };

    let src_corners: Vector<Point2f> = Vector::from_iter([
        Point2f::new(0.0, 0.0),
        Point2f::new(w_img, 0.0),
        Point2f::new(w_img, h_img),
        Point2f::new(0.0, h_img),
    ]);

    let dst_corners: Vector<Point2f> = Vector::from_iter([
        project_point((0.0, 0.0)),
        project_point((w_img, 0.0)),
        project_point((w_img, h_img)),
        project_point((0.0, h_img)),
    ]);

    let h = opencv::imgproc::get_perspective_transform(
        &src_corners,
        &dst_corners,
        opencv::core::DECOMP_LU,
    )?;

    Ok(Some((h, cm_per_pixel)))
}

//--------------------------------------------------------------------------------------------------

fn order_points(pts: &[(f32, f32)]) -> Vec<(f32, f32)> {
    let n = pts.len() as f32;
    let cx = pts.iter().map(|p| p.0).sum::<f32>() / n;
    let cy = pts.iter().map(|p| p.1).sum::<f32>() / n;

    let mut sorted: Vec<(f32, (f32, f32))> = pts
        .iter()
        .map(|&(x, y)| ((y - cy).atan2(x - cx), (x, y)))
        .collect();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let sorted: Vec<(f32, f32)> = sorted.into_iter().map(|(_, p)| p).collect();

    let start = sorted
        .iter()
        .enumerate()
        .min_by(|a, b| (a.1.0 + a.1.1).partial_cmp(&(b.1.0 + b.1.1)).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    (0..sorted.len()).map(|i| sorted[(start + i) % sorted.len()]).collect()
}

//--------------------------------------------------------------------------------------------------

pub fn clean_double_dot(price: &str) -> String {
    let chars: Vec<char> = price.chars().collect();
    let mut result = String::new();

    for i in 0..chars.len() {
        let is_digit = chars[i].is_ascii_digit();
        let next_is_digit = i + 1 < chars.len() && chars[i + 1].is_ascii_digit();
        if !is_digit && !next_is_digit {
            continue;
        }
        result.push(chars[i]);
    }
    result
}

//--------------------------------------------------------------------------------------------------

pub fn clean_thousand_dot(price: &str) -> String {
    let chars: Vec<char> = price.chars().collect();
    let mut reversed: Vec<char> = Vec::new();
    let mut counter_dot = 0;

    for i in (0..chars.len()).rev() {
        if chars[i] == '.' {
            counter_dot += 1;
            if counter_dot > 1 {
                continue;
            }
        }
        reversed.push(chars[i]);
    }
    reversed.reverse();
    let final_price: String = reversed.into_iter().collect();

    let price_parts: Vec<&str> = final_price.split('.').collect();
    if price_parts.len() == 2 {
        let frac = price_parts[1];
        match frac.len() {
            0..=2 => final_price,
            3..=4 => price_parts.concat(),
            5 => format!("{}{}.{}", price_parts[0], &frac[..3], &frac[3..]),
            _ => final_price,
        }
    } else {
        final_price
    }
}

//--------------------------------------------------------------------------------------------------
// End of file
//--------------------------------------------------------------------------------------------------