use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use numpy::PyReadonlyArray3;
use opencv::core::{Mat, CV_8UC3};
use opencv::prelude::*;

use crate::models::model_core::ModelPipeline;
use crate::models::cnn_digit::CNNDigit;

#[pyclass(name = "CNNDigit")]
pub struct PyCNNDigit {
    inner: CNNDigit,
}

#[pymethods]
impl PyCNNDigit {
    #[new]
    fn new() -> Self {
        PyCNNDigit { inner: CNNDigit::new() }
    }

    fn setup_model(&mut self, model_path: &str, device: &str) -> PyResult<()> {
        self.inner
            .setup_model(model_path, device)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Prend une image numpy (H, W, 3) en uint8, renvoie (chiffre, confiance)
    fn predict(&mut self, image: PyReadonlyArray3<'_, u8>) -> PyResult<(usize, f32)> {
        let vue = image.as_array();
        let shape = vue.shape();
        let (hauteur, largeur, canaux) = (shape[0], shape[1], shape[2]);

        if canaux != 3 {
            return Err(PyRuntimeError::new_err(
                "L'image doit avoir 3 canaux (RGB/BGR)",
            ));
        }

        // Copie en Vec<u8> contigu (nécessaire pour construire le Mat OpenCV)
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

        // On clone en Mat "owned" car `donnees` va être libéré à la fin du scope
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

        Ok((resultat.digit, resultat.confidence))
    }
}
