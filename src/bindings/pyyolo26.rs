use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use numpy::PyReadonlyArray3;
use opencv::core::{Mat, CV_8UC3};
use opencv::prelude::*;

use crate::models::model_core::ModelPipeline;
use crate::models::yolo_26::YOLO26;
use crate::detections::detections::Detection;
use crate::detections::bbox::BBox;
use crate::detections::mask::Mask;

//--------------------------------------------------------------------------------------------------
// Wrapper du modèle
//--------------------------------------------------------------------------------------------------

#[pyclass(name = "Yolo26")]
pub struct PyYolo26 {
    inner: YOLO26,
}

#[pymethods]
impl PyYolo26 {
    #[new]
    fn new() -> Self {
        PyYolo26 { inner: YOLO26::new() }
    }

    fn setup_model(&mut self, model_path: &str, device: &str) -> PyResult<()> {
        self.inner
            .setup_model(model_path, device)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Prend une image numpy (H, W, 3) en uint8, renvoie une liste de détections
    fn predict(&mut self, image: PyReadonlyArray3<'_, u8>) -> PyResult<Vec<Detection>> {
        let vue = image.as_array();
        let shape = vue.shape();
        let (hauteur, largeur, canaux) = (shape[0], shape[1], shape[2]);

        if canaux != 3 {
            return Err(PyRuntimeError::new_err(
                "L'image doit avoir 3 canaux (RGB/BGR)",
            ));
        }

        let donnees: Vec<u8> = vue.iter().copied().collect();

        let mat = unsafe {
            Mat::new_rows_cols_with_data_unsafe(
                hauteur as i32,
                largeur as i32,
                CV_8UC3,
                donnees.as_ptr() as *mut std::ffi::c_void,
                opencv::core::Mat_AUTO_STEP,
            )
        }
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let mat = mat
            .try_clone()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let tensor = self
            .inner
            .preprocess(&mat)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let sortie_brute = self
            .inner
            .infer(&tensor)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let resultat = self
            .inner
            .postprocess(&sortie_brute)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        // Conversion Vec<Detection> (Rust interne) -> Vec<PyDetection> (exposé Python)
        let detections_py: Vec<Detection> = resultat
            .detections
            .into_iter()
            .map(|d| {
                let categorie = d.class_label
                    .parse()
                    .map_err(|e: String| PyRuntimeError::new_err(e))?;

                let mask_mat = mat.try_clone()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

                let bbox = BBox::new(
                    d.bbox.x1,
                    d.bbox.y1,
                    d.bbox.x2,
                    d.bbox.y2,
                    (hauteur, largeur),
                );

                let mask = Mask {
                    mat: mask_mat,
                    _mat_bin: Mat::default(),
                };

                Ok(Detection::new(categorie, bbox, mask, "".to_string()))
            })
            .collect::<Result<Vec<_>, PyErr>>()?;

        Ok(detections_py)
    }
}