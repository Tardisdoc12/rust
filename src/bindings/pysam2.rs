use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use numpy::PyReadonlyArray3;
use opencv::core::{Mat, CV_8UC3};
use opencv::prelude::*;

use crate::models::sam2::Sam2Processor;
use crate::detections::mask::Mask;

//--------------------------------------------------------------------------------------------------
// Wrapper du modèle
//--------------------------------------------------------------------------------------------------

#[pyclass(name = "Sam2Processor")]
pub struct PySam2Processor {
    inner: Sam2Processor,
}

#[pymethods]
impl PySam2Processor {
    #[new]
    fn new() -> Self {
        PySam2Processor { inner: Sam2Processor::new() }
    }

    fn setup_model(&mut self, encoder_path: &str, decoder_path: &str, device: &str) -> PyResult<()> {
        self.inner
            .setup_model(encoder_path, decoder_path, device)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn setup_image(&mut self, image: PyReadonlyArray3<'_, u8>) -> PyResult<()> {
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

        self.inner
            .set_image(&mat)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(())
    }

    fn predict_box(&mut self, bbox: (f32, f32, f32, f32)) -> PyResult<Mask> {
        let resultat = self
            .inner
            .predict_box(bbox)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(resultat)

    }
}