/// Classification result structure and utility functions.

#[derive(Debug)]
#[allow(dead_code)]
pub struct ClassifResult {
    pub class_label: String,
    pub score: f32,
    pub margin: f32,   // p_top1 - p_top2
    pub energy: f32,
}

pub fn top2_margin(probs: &[f32]) -> f32 {
    let (mut p1, mut p2) = (0.0f32, 0.0f32);
    for &p in probs {
        if p > p1 { p2 = p1; p1 = p; }
        else if p > p2 { p2 = p; }
    }
    p1 - p2
}