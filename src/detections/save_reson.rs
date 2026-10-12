use crate::detections::classify_result::ClassifResult;

const MARGIN_THR_INCEPTION: f32 = 0.60;
const MARGIN_THR_EFFICIENT: f32 = 0.60;
const RANDOM_RATE: f32 = 0.02;

#[derive(Clone)]
pub enum SaveReason { Disagreement, LowMargin, RandomConfident }

impl SaveReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Disagreement => "disagreement",
            Self::LowMargin => "low_margin",
            Self::RandomConfident => "random_confident",
        }
    }
}

pub fn should_save(inc: &ClassifResult, eff: &ClassifResult) -> Option<SaveReason> {
    if inc.class_label != eff.class_label {
        return Some(SaveReason::Disagreement);
    }
    if inc.margin < MARGIN_THR_INCEPTION || eff.margin < MARGIN_THR_EFFICIENT {
        return Some(SaveReason::LowMargin);
    }
    if rand::random::<f32>() < RANDOM_RATE {
        return Some(SaveReason::RandomConfident);
    }
    None
}
