use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use numpy::PyReadonlyArray3;
use opencv::core::{Mat, CV_8UC3};
use opencv::prelude::*;

use crate::models::model_core::ModelPipeline;
use crate::models::yolo_26::YOLO26;

//--------------------------------------------------------------------------------------------------
// Types de sortie exposés à Python
//--------------------------------------------------------------------------------------------------

#[pyclass(name = "BoundingBox")]
#[derive(Clone)]
pub struct PyBoundingBox {
    #[pyo3(get)]
    pub x1: f32,
    #[pyo3(get)]
    pub y1: f32,
    #[pyo3(get)]
    pub x2: f32,
    #[pyo3(get)]
    pub y2: f32,
}

#[pymethods]
impl PyBoundingBox {
    fn __repr__(&self) -> String {
        format!(
            "BoundingBox(x1={:.1}, y1={:.1}, x2={:.1}, y2={:.1})",
            self.x1, self.y1, self.x2, self.y2
        )
    }
}

#[pyclass(name = "Detection")]
pub struct PyDetection {
    #[pyo3(get)]
    pub bbox: PyBoundingBox,
    #[pyo3(get)]
    pub score: f32,
    #[pyo3(get)]
    pub class_id: usize,
    #[pyo3(get)]
    pub class_label: String,
}

#[pymethods]
impl PyDetection {
    fn __repr__(&self) -> String {
        format!(
            "Detection(label='{}', score={:.2}, bbox={})",
            self.class_label, self.score, self.bbox.__repr__()
        )
    }
}

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
    fn predict(&mut self, image: PyReadonlyArray3<'_, u8>) -> PyResult<Vec<PyDetection>> {
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
        let detections_py = resultat
            .detections
            .into_iter()
            .map(|d| PyDetection {
                bbox: PyBoundingBox {
                    x1: d.bbox.x1,
                    y1: d.bbox.y1,
                    x2: d.bbox.x2,
                    y2: d.bbox.y2,
                },
                score: d.score,
                class_id: d.class_id,
                class_label: d.class_label,
            })
            .collect();

        Ok(detections_py)
    }
}