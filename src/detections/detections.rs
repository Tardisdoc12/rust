//--------------------------------------------------------------------------------------------------
// Filename: detections.rs
// Author: Jean Anquetil
// Date: 2026-09-07
//--------------------------------------------------------------------------------------------------

use pyo3::prelude::*;
use opencv::core::Mat;

use crate::detections::bbox::BBox;
use crate::detections::mask::Mask;
use numpy::PyReadonlyArray3;

use crate::detections::detection_class::DetectionClass;

//--------------------------------------------------------------------------------------------------

#[pyclass(name = "Detection", from_py_object)]
#[derive(Clone)]
pub struct Detection {
    #[pyo3(get, set)]
    pub categorie: DetectionClass,
    #[pyo3(get, set)]
    pub bbox: BBox,
    #[pyo3(get, set)]
    pub score: f32,
    #[pyo3(get, set)]
    pub label: String,
    #[pyo3(get, set)]
    pub mask: Mask,
    #[pyo3(get, set)]
    pub base_name: String,
    #[pyo3(get, set)]
    pub price: f32,
    #[pyo3(get, set)]
    pub width_cm: f32,
    #[pyo3(get, set)]
    pub height_cm: f32,
    #[pyo3(get, set)]
    pub shelf_position: i32,
    #[pyo3(get, set)]
    pub position: i32,
}

//--------------------------------------------------------------------------------------------------

impl Detection {
    fn setup_mat(&mut self, mat: Mat) {
        self.mask.setup_mat(mat);
    }

    pub fn crop_mask(&self)-> anyhow::Result<Mat> {
        let (x1, y1, x2, y2) = self.bbox.xyxy();
        self.mask.get_subpart_mat((
            x1 as i32,
            y1 as i32,
            x2 as i32,
            y2 as i32,
        ))
    }

    pub fn get_real_size_from_homography(&self, h: &Mat) -> anyhow::Result<(f32, f32)> {
        self.mask.get_real_size_from_homography(h)
    }

    pub fn set_shelf_position(&mut self, etiquettes_centers: &[f32]) {
        if etiquettes_centers.is_empty() {
            self.shelf_position = 100;
            return;
        }

        let nbr_ligne_etiquette = etiquettes_centers.len().saturating_sub(1);
        let center_y = self.bbox.center_rel().1;

        for iter_ett in 0..nbr_ligne_etiquette {
            let upper_bound = etiquettes_centers[iter_ett + 1];
            let lower_bound = etiquettes_centers[iter_ett];

            if center_y < upper_bound && center_y > lower_bound {
                self.shelf_position = (iter_ett + 1) as i32;
            } else if center_y == upper_bound || center_y == lower_bound {
                self.shelf_position = iter_ett as i32;
            }
        }

        if center_y < etiquettes_centers.iter().copied().reduce(f32::max).unwrap_or(f32::MAX) {
            self.shelf_position = nbr_ligne_etiquette as i32 + 1;
        }
    }


}

#[pymethods]
impl Detection {
    #[new]
    pub fn new(
        categorie: DetectionClass,
        bbox: BBox,
        mask: Mask,
        base_name: String,
    ) -> Self {
        Self {
            categorie,
            bbox,
            score : 0.0,
            label : "".to_string(),
            mask: mask,
            base_name,
            price : 0.0,
            width_cm: 0.0,
            height_cm: 0.0,
            shelf_position: 0,
            position: 0,
        }
    }

    fn setup_mask_from_python(&mut self, image: PyReadonlyArray3<'_, u8>) -> PyResult<()> {
        self.mask.setup_from_python(image)
    }

    fn set_prediction(&mut self, score: f32, label: String) {
        self.score = score;
        self.label = label;
    }

    fn set_size(&mut self, width_cm: f32, height_cm: f32) {
        self.width_cm = width_cm;
        self.height_cm = height_cm;
    }
}

//--------------------------------------------------------------------------------------------------
// End of file
//--------------------------------------------------------------------------------------------------