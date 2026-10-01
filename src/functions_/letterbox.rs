//--------------------------------------------------------------------------------------------------
// Filename: letterbox.rs
// Author: Jean Anquetil
// Date: 2026-10-01
//--------------------------------------------------------------------------------------------------

use opencv::core::{self, Mat, Scalar, Size};
use opencv::prelude::*;
use opencv::imgproc;

//--------------------------------------------------------------------------------------------------
/// Redimensionne `image` (BGR) pour qu'elle tienne dans `target` x `target`
/// en gardant le ratio, la centre, et remplit le reste avec `fill_rgb`.
/// Renvoie une Mat RGB de taille target x target.

pub fn letterbox_rgb(image: &Mat, target: i32, fill_rgb: [u8; 3]) -> anyhow::Result<Mat> {
    let (w, h) = (image.cols(), image.rows());
    if w <= 0 || h <= 0 {
        anyhow::bail!("Image vide ({}x{})", w, h);
    }

    let scale = target as f64 / w.max(h) as f64;
    let new_w = ((w as f64 * scale).round() as i32).clamp(1, target);
    let new_h = ((h as f64 * scale).round() as i32).clamp(1, target);

    // INTER_AREA donne de meilleurs résultats quand on réduit
    let interp = if scale < 1.0 { imgproc::INTER_AREA } else { imgproc::INTER_LINEAR };

    let mut resized = Mat::default();
    imgproc::resize(image, &mut resized, Size::new(new_w, new_h), 0.0, 0.0, interp)?;

    // BGR -> RGB avant le padding, pour que `fill_rgb` soit bien en RGB
    let mut rgb = Mat::default();
    imgproc::cvt_color(&resized, &mut rgb, imgproc::COLOR_BGR2RGB, 0)?;

    let pad_w = target - new_w;
    let pad_h = target - new_h;
    let (top, left) = (pad_h / 2, pad_w / 2);
    let (bottom, right) = (pad_h - top, pad_w - left);

    let mut out = Mat::default();
    core::copy_make_border(
        &rgb, &mut out,
        top, bottom, left, right,
        core::BORDER_CONSTANT,
        Scalar::new(fill_rgb[0] as f64, fill_rgb[1] as f64, fill_rgb[2] as f64, 0.0),
    )?;
    Ok(out)
}

//--------------------------------------------------------------------------------------------------
// End of file
//--------------------------------------------------------------------------------------------------