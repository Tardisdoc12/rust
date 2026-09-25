//--------------------------------------------------------------------------------------------------
// Filename: pipeline.rs
// Author: Jean Anquetil
// Date: 2026-09-08
//--------------------------------------------------------------------------------------------------

use std::sync::Arc;

use crate::pipeline::pipeline_data::PipelineData;

//--------------------------------------------------------------------------------------------------

pub type StepFn = Box<dyn FnMut(PipelineData) -> PipelineData>;

//--------------------------------------------------------------------------------------------------

pub enum PipelineStep {
    Single(StepFn),
    Branch(Vec<StepFn>),
}

//--------------------------------------------------------------------------------------------------

pub struct Pipeline {
    steps: Vec<PipelineStep>,
}

//--------------------------------------------------------------------------------------------------

impl Pipeline {
    pub fn new(steps: Vec<PipelineStep>) -> Self {
        Self { steps }
    }

    pub fn add_step(&mut self, step: PipelineStep) {
        self.steps.push(step);
    }

    pub fn run(&mut self, input: PipelineData) -> PipelineData {
        let mut output = input;

        for step in self.steps.iter_mut() {
            output = match step {
                PipelineStep::Single(f) => f(output),

                PipelineStep::Branch(fs) => {
                    let results = fs
                        .iter_mut()
                        .map(|f| f(output.clone())) // clone = juste Arc::clone, pas cher
                        .collect();
                    PipelineData::Multiple(Arc::new(results))
                }
            };
        }

        output
    }
}

//--------------------------------------------------------------------------------------------------
// End of file
//--------------------------------------------------------------------------------------------------