//--------------------------------------------------------------------------------------------------
// Filename: model_core.rs
// Author: Jean Anquetil
// Date: 2026-08-24
//--------------------------------------------------------------------------------------------------

use opencv::core::Mat;

//--------------------------------------------------------------------------------------------------

pub trait ModelPipeline {
    type Output;

    fn setup_model(&mut self, model_path: &str, device: &str) -> anyhow::Result<()>;
    fn preprocess(&self, image: &Mat) -> anyhow::Result<Vec<f32>>; // tensor aplati, prêt pour ort
    fn infer(&mut self, input: &[f32]) -> anyhow::Result<Vec<f32>>;
    fn postprocess(&self, raw_output: &[f32]) -> anyhow::Result<Self::Output>;

    // fn process(&mut self, image_path: &str) -> anyhow::Result<Self::Output> {
    //     let img = opencv::imgcodecs::imread(image_path, opencv::imgcodecs::IMREAD_COLOR)?;
    //     let input = self.preprocess(&img)?;
    //     let raw = self.infer(&input)?;
    //     self.postprocess(&raw)
    // }
}

//--------------------------------------------------------------------------------------------------
// End of file
//--------------------------------------------------------------------------------------------------