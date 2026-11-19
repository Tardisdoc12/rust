//--------------------------------------------------------------------------------------------------
// Filename: florence.rs
// Author: Jean Anquetil
// Date: Thursday 19/11/2026
//--------------------------------------------------------------------------------------------------

use opencv::{core::{Mat, Rect, Scalar, Size, CV_32F, CV_8UC3}, imgproc, prelude::*};
use ort::{session::Session, value::Tensor};
use serde::Deserialize;
use tokenizers::Tokenizer;
use crate::models::model_core::ModelPipeline;
use crate::functions_::setup_model_avec_cache::setup_model_avec_cache;

#[derive(Deserialize, Clone)]
struct Gen {
    img_size: usize, square: bool, mean: [f32; 3], std: [f32; 3],
    decoder_start_token_id: i64, eos_token_id: i64, pad_token_id: i64,
    forced_bos_token_id: Option<i64>, #[serde(default)] forced_eos_token_id: Option<i64>,
    max_new_tokens: usize, #[serde(default)] no_repeat_ngram_size: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct PriceRead { pub text: String, pub price: Option<f64>, pub conf: f32 }

pub struct Florence { enc: Option<Session>, dec: Option<Session>, tok: Option<Tokenizer>, g: Option<Gen> }
impl Florence { pub fn new() -> Self { Self { enc: None, dec: None, tok: None, g: None } } }

/// Rend le crop carré sans le déformer : fond = moyenne (tronquée) des pixels du pourtour, image centrée.
fn pad_square(rgb: &Mat) -> anyhow::Result<Mat> {
    let (w, h) = (rgb.cols(), rgb.rows());
    if w == h { return Ok(rgb.try_clone()?); }
    let px = rgb.data_bytes()?;                                   // RGB contigu
    let at = |y: i32, x: i32, c: usize| px[((y * w + x) * 3) as usize + c] as u64;
    let mut sum = [0u64; 3]; let mut n = 0u64;
    for x in 0..w { for &y in &[0, h - 1] { for c in 0..3 { sum[c] += at(y, x, c); } n += 1; } }
    for y in 0..h { for &x in &[0, w - 1] { for c in 0..3 { sum[c] += at(y, x, c); } n += 1; } }
    let fill = Scalar::new((sum[0] / n) as f64, (sum[1] / n) as f64, (sum[2] / n) as f64, 0.0);   // int(mean) comme Python
    let s = w.max(h);
    let mut canvas = Mat::new_rows_cols_with_default(s, s, CV_8UC3, fill)?;
    let mut roi = Mat::roi_mut(&mut canvas, Rect::new((s - w) / 2, (s - h) / 2, w, h))?;
    rgb.copy_to(&mut roi)?;
    Ok(canvas)
}

impl ModelPipeline for Florence {
    type Output = PriceRead;

    /// model_path = DOSSIER du modèle exporté.
    fn setup_model(&mut self, model_path: &str, device: &str) -> anyhow::Result<()> {
        let d = std::path::Path::new(model_path);
        self.g = Some(serde_json::from_slice(&std::fs::read(d.join("gen.json"))?)?);
        self.tok = Some(Tokenizer::from_file(d.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("{e}"))?);
        self.enc = Some(setup_model_avec_cache(&d.join("encoder.onnx").to_string_lossy(), device)?);
        self.dec = Some(setup_model_avec_cache(&d.join("decoder.onnx").to_string_lossy(), device)?);
        Ok(())
    }

    fn preprocess(&self, image: &Mat) -> anyhow::Result<Vec<f32>> {
        let g = self.g.as_ref().ok_or_else(|| anyhow::anyhow!("modèle non chargé"))?;
        let mut rgb = Mat::default();
        match image.channels() {
            3 => imgproc::cvt_color(image, &mut rgb, imgproc::COLOR_BGR2RGB, 0)?,
            4 => imgproc::cvt_color(image, &mut rgb, imgproc::COLOR_BGRA2RGB, 0)?,   // pas de composition sur blanc : crops YOLO sans alpha
            1 => imgproc::cvt_color(image, &mut rgb, imgproc::COLOR_GRAY2RGB, 0)?,
            n => anyhow::bail!("canaux non gérés: {n}"),
        }
        let sq = if g.square { pad_square(&rgb)? } else { rgb };
        let s = g.img_size as i32;
        let interp = if sq.cols() >= s && sq.rows() >= s { imgproc::INTER_AREA } else { imgproc::INTER_CUBIC };  // PIL bicubique est antialiasé
        let mut r = Mat::default();
        imgproc::resize(&sq, &mut r, Size::new(s, s), 0.0, 0.0, interp)?;
        let mut f = Mat::default();
        r.convert_to(&mut f, CV_32F, 1.0 / 255.0, 0.0)?;
        let px = f.data_typed::<f32>()?;                         // HWC
        let n = g.img_size * g.img_size;
        let mut chw = vec![0f32; 3 * n];
        for i in 0..n { for c in 0..3 { chw[c * n + i] = (px[i * 3 + c] - g.mean[c]) / g.std[c]; } }
        Ok(chw)
    }

    /// Une image par appel. Renvoie [id, logp] pour chaque jeton généré (eos inclus).
    fn infer(&mut self, input: &[f32]) -> anyhow::Result<Vec<f32>> {
        let g = self.g.clone().ok_or_else(|| anyhow::anyhow!("modèle non chargé"))?;
        let sz = g.img_size;
        anyhow::ensure!(input.len() == 3 * sz * sz, "taille d'entrée {} != {}", input.len(), 3 * sz * sz);
        let enc = self.enc.as_mut().ok_or_else(|| anyhow::anyhow!("modèle non chargé"))?;
        let out = enc.run(ort::inputs!["pixel_values" => Tensor::from_array(([1usize, 3, sz, sz], input.to_vec().into_boxed_slice()))?])?;
        let (shape, data) = out["encoder_hidden_states"].try_extract_tensor::<f32>()?;
        let (l, h) = (shape[1] as usize, shape[2] as usize);
        let ehs = data.to_vec();
        drop(out);

        // Même règles que HF generate + Runner.read (vérifiées en Python, cf. export_florence_onnx.ort_read) :
        //  - jeton forcé en 1re position, compté dans la confiance avec log-proba 0 et dans max_new_tokens ;
        //  - no_repeat_ngram_size : jetons qui répéteraient un n-gramme interdits AVANT la normalisation ;
        //  - EOS forcé (log-proba 0) au dernier pas autorisé.
        let mut seq: Vec<i64> = vec![g.decoder_start_token_id];
        let mut raw: Vec<f32> = Vec::new();
        let mut steps = g.max_new_tokens;
        if let Some(b) = g.forced_bos_token_id { seq.push(b); raw.push(b as f32); raw.push(0.0); steps -= 1; }
        let dec = self.dec.as_mut().ok_or_else(|| anyhow::anyhow!("modèle non chargé"))?;
        for step in 0..steps {
            let (best, lp) = if g.forced_eos_token_id.is_some() && step == steps - 1 {
                (g.forced_eos_token_id.unwrap() as usize, 0.0f64)
            } else {
                let o = dec.run(ort::inputs![
                    "decoder_input_ids" => Tensor::from_array(([1usize, seq.len()], seq.clone().into_boxed_slice()))?,
                    "encoder_hidden_states" => Tensor::from_array(([1usize, l, h], ehs.clone().into_boxed_slice()))?,
                ])?;
                let (_, lg) = o["logits"].try_extract_tensor::<f32>()?;     // [1, V]
                let banned = banned_ngram(&seq, g.no_repeat_ngram_size.unwrap_or(0));
                let (mut best, mut bv) = (0usize, f32::NEG_INFINITY);
                for (i, &v) in lg.iter().enumerate() {
                    if banned.contains(&(i as i64)) { continue; }
                    if v > bv { bv = v; best = i; }
                }
                let lse = bv as f64 + lg.iter().enumerate().filter(|(i, _)| !banned.contains(&(*i as i64)))
                    .map(|(_, &v)| ((v - bv) as f64).exp()).sum::<f64>().ln();
                (best, bv as f64 - lse)
            };
            raw.push(best as f32); raw.push(lp as f32);
            seq.push(best as i64);
            if best as i64 == g.eos_token_id { break; }
        }
        Ok(raw)
    }

    fn postprocess(&self, raw: &[f32]) -> anyhow::Result<PriceRead> {
        let tok = self.tok.as_ref().ok_or_else(|| anyhow::anyhow!("modèle non chargé"))?;
        let ids: Vec<u32> = raw.chunks(2).map(|c| c[0] as u32).collect();
        let lps: Vec<f64> = raw.chunks(2).map(|c| c[1] as f64).collect();
        let text = tok.decode(&ids, true).map_err(|e| anyhow::anyhow!("{e}"))?.trim().to_string();
        let conf = if lps.is_empty() { 0.0 } else { (lps.iter().sum::<f64>() / lps.len() as f64).exp() as f32 };
        Ok(PriceRead { price: to_price(&text), text, conf })
    }

    fn unload(&mut self) { self.enc = None; self.dec = None; self.tok = None; self.g = None; }
}

/// Jetons interdits par no_repeat_ngram_size = n (règle de NoRepeatNGramLogitsProcessor).
fn banned_ngram(seq: &[i64], n: usize) -> Vec<i64> {
    if n == 0 || seq.len() < n { return Vec::new(); }
    let prefix = &seq[seq.len() - (n - 1)..];
    (0..=seq.len() - n).filter(|&i| &seq[i..i + n - 1] == prefix).map(|i| seq[i + n - 1]).collect()
}

/// sortie du modèle ("2899.00" / "-") -> prix. Même règle que from_target Python pour le cas normal (repli simplifié).
fn to_price(text: &str) -> Option<f64> {
    let s = text.trim();
    if matches!(s.to_lowercase().as_str(), "-" | "" | "aucun" | "none") || !s.chars().any(|c| c.is_ascii_digit()) { return None; }
    s.replace(',', ".").parse::<f64>().ok().filter(|v| *v > 0.0)
}

impl Florence {
    /// image -> lecture complète (les trois étapes).
    pub fn read(&mut self, image: &Mat) -> anyhow::Result<PriceRead> {
        let x = self.preprocess(image)?;
        let raw = self.infer(&x)?;
        self.postprocess(&raw)
    }
}

//--------------------------------------------------------------------------------------------------
// END OF FILE
//--------------------------------------------------------------------------------------------------