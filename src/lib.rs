use pyo3::prelude::*;

mod detections;
mod models;
mod processor;
mod functions_;
mod bindings;
mod tools_class;
use detections::detection_class::DetectionClass;
use detections::detections::Detection;
use detections::bbox::BBox;
use detections::mask::Mask;
use bindings::pycnndigit::PyCNNDigit;
use bindings::pyyolo26::PyYolo26;
use bindings::pysam2::PySam2Processor;
use bindings::pydetectionpipeline::PyDetectionPipeline;

#[pymodule]
fn rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCNNDigit>()?;
    m.add_class::<PyYolo26>()?;
    m.add_class::<PySam2Processor>()?;
    m.add_class::<Detection>()?;
    m.add_class::<DetectionClass>()?;
    m.add_class::<BBox>()?;
    m.add_class::<Mask>()?;
    m.add_class::<PyDetectionPipeline>()?;
    Ok(())
}