use pyo3::prelude::*;

// mod models;
// mod processor;
// mod functions_;
// use models::inceptionv3;
// use models::efficientnetb2;
// use models::yolo_v7;
// use models::yolo_26;
// use models::cnn_digit;
// use models::sam2;
// use processor::processor;
mod detections;
mod models;
mod processor;
mod functions_;
mod bindings;
mod pipeline;
mod tools_class;
use detections::detection_class::DetectionClass;
use detections::detections::Detection;
use detections::bbox::BBox;
use detections::mask::Mask;
use bindings::pycnndigit::PyCNNDigit;
use bindings::pyyolo26::PyYolo26;
use bindings::pysam2::PySam2Processor;
use bindings::pydetectionpipeline::PyDetectionPipeline;
use pipeline::workflow;



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
    m.add_function(wrap_pyfunction!(crate::pipeline::workflow::workflows, m)?)?;
    Ok(())
}

// #[pymodule]
// fn mon_module(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
//     let sys_modules = py.import("sys")?.getattr("modules")?;

//     // Liste de (nom, fonction constructrice) — ajoute simplement une ligne
//     // ici pour chaque nouveau sous-module, plus besoin de dupliquer le reste
//     let sous_modules: [(&str, fn(Python<'_>) -> PyResult<Bound<'_, PyModule>>); 2] = [
//         ("inception_v3", inceptionv3::module),
//         ("efficientnet_b2", efficientnetb2::module),
//         ("yolo_v7", yolo_v7::module),
//         ("yolo_26", yolo_26::module),
//         ("cnn_digit", cnn_digit::module),
//         ("sam2", sam2::module),
//         ("processor", processor::module),
//     ];

//     for (nom, constructeur) in sous_modules {
//         let sous_mod = constructeur(py)?;
//         m.add_submodule(&sous_mod)?;
//         sys_modules.set_item(format!("mon_module.{nom}"), sous_mod)?;
//     }

//     Ok(())
// }