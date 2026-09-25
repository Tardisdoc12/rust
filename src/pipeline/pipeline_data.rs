//--------------------------------------------------------------------------------------------------
// Filename: pipeline_data.rs
// Author: Jean Anquetil
// Date: 2026-09-10
//--------------------------------------------------------------------------------------------------

use std::sync::Arc;
use crate::detections::mask::Mask;
use crate::detections::detections::Detection;

#[derive(Clone)]
pub enum PipelineData {
    Image(Arc<Mask>),
    Detections(Arc<Vec<Detection>>),
    Multiple(Arc<Vec<PipelineData>>),
}

//--------------------------------------------------------------------------------------------------
// End of file
//--------------------------------------------------------------------------------------------------