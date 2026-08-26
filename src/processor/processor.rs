//--------------------------------------------------------------------------------------------------
// Filename: processor.rs
// Author: Jean Anquetil
// Date: 2026-08-24
//--------------------------------------------------------------------------------------------------

use crate::models::model_core::ModelPipeline;

//--------------------------------------------------------------------------------------------------

pub struct Processor<M>
where
    M: ModelPipeline,
{
    pub model: M,
}

impl<M> Processor<M>
where
    M: ModelPipeline,
{
    pub fn new(mut model: M, model_path: &str, device: &str) -> anyhow::Result<Self> {
        model.setup_model(model_path, device)?;
        Ok(Self { model })
    }

    pub fn process(&mut self, image: &opencv::core::Mat) -> anyhow::Result<M::Output> {
        let input = self.model.preprocess(image)?;   // Mat -> Vec<f32>
        let raw = self.model.infer(&input)?;          // Vec<f32> -> Vec<f32> (logits bruts)
        let output = self.model.postprocess(&raw)?;   // Vec<f32> -> Self::Output
        Ok(output)
    }
}

//--------------------------------------------------------------------------------------------------
// End of file
//--------------------------------------------------------------------------------------------------