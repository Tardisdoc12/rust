//--------------------------------------------------------------------------------------------------
// Filename: qwen_onnx.rs
// Author: Jean Anquetil
// Date: Wednesday 18/11/2026
//--------------------------------------------------------------------------------------------------

use std::collections::HashMap;
use opencv::{core::{Mat, Size, CV_32F}, imgproc, prelude::*};
use ort::{session::Session, value::Tensor};
use tokenizers::Tokenizer;

use crate::models::vlm::{Part, Prompt, VlmBackend, LETTERS, NONE_LETTER};
use crate::functions_::setup_model_avec_cache::setup_model_avec_cache;

//--------------------------------------------------------------------------------------------------

const FACTOR: i32 = 28;           // patch 14 x merge 2
const P: usize = 14;
const MERGE: usize = 2;
const MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

//--------------------------------------------------------------------------------------------------

/// smart_resize de Qwen2-VL : multiples de 28, pixels entre min et max.
fn smart_resize(h: i32, w: i32, min_pixels: i32, max_pixels: i32) -> (i32, i32) {
    let r = |x: f64| (x / FACTOR as f64).round_ties_even() as i32 * FACTOR;
    let (mut hb, mut wb) = (r(h as f64).max(FACTOR), r(w as f64).max(FACTOR));
    if (hb * wb) as i64 > max_pixels as i64 {
        let beta = ((h as f64 * w as f64) / max_pixels as f64).sqrt();
        hb = (((h as f64 / beta) / FACTOR as f64).floor() as i32 * FACTOR).max(FACTOR);
        wb = (((w as f64 / beta) / FACTOR as f64).floor() as i32 * FACTOR).max(FACTOR);
    } else if (hb * wb) < min_pixels {
        let beta = (min_pixels as f64 / (h as f64 * w as f64)).sqrt();
        hb = ((h as f64 * beta / FACTOR as f64).ceil() as i32) * FACTOR;
        wb = ((w as f64 * beta / FACTOR as f64).ceil() as i32) * FACTOR;
    }
    (hb, wb)
}

//--------------------------------------------------------------------------------------------------

/// Mat BGR -> (patches [gh*gw, 1176], gh, gw) dans l'ordre du Qwen2VLImageProcessor (blocs de fusion 2x2 contigus).
fn patchify(img: &Mat, min_pixels: i32, max_pixels: i32) -> anyhow::Result<(Vec<f32>, usize, usize)> {
    let (h, w) = smart_resize(img.rows(), img.cols(), min_pixels, max_pixels);
    let mut rgb = Mat::default();
    imgproc::cvt_color(img, &mut rgb, imgproc::COLOR_BGR2RGB, 0)?;
    let interp = if img.rows() >= h && img.cols() >= w { imgproc::INTER_AREA } else { imgproc::INTER_CUBIC };
    let mut r = Mat::default();
    imgproc::resize(&rgb, &mut r, Size::new(w, h), 0.0, 0.0, interp)?;
    let mut f = Mat::default();
    r.convert_to(&mut f, CV_32F, 1.0 / 255.0, 0.0)?;
    let px = f.data_typed::<f32>()?;                       // HWC
    let (h, w) = (h as usize, w as usize);
    let (gh, gw) = (h / P, w / P);
    let (a_n, b_n) = (gh / MERGE, gw / MERGE);
    let mut out = vec![0f32; gh * gw * 3 * 2 * P * P];
    for a in 0..a_n { for b in 0..b_n { for mh in 0..MERGE { for mw in 0..MERGE {
        let row = ((a * b_n + b) * MERGE + mh) * MERGE + mw;
        for c in 0..3 { for ph in 0..P { for pw in 0..P {
            let y = (a * MERGE + mh) * P + ph; let x = (b * MERGE + mw) * P + pw;
            let v = (px[(y * w + x) * 3 + c] - MEAN[c]) / STD[c];
            for t in 0..2 {                                 // image fixe : les 2 images temporelles sont identiques
                out[row * 1176 + ((c * 2 + t) * P + ph) * P + pw] = v;
            }
        }}}
    }}}}
    Ok((out, gh, gw))
}

//--------------------------------------------------------------------------------------------------

/// Positions (3,S) = port du get_rope_index du script officiel. grids = (gh, gw) AVANT fusion 2x2 de chaque image.
/// `time_scale` = tokens_per_second (config.vision_config.tokens_per_second) x second_per_grid (1 dans le script).
fn rope_positions(ids: &[i64], image_pad: i64, grids: &[(usize, usize)], time_scale: i64) -> Vec<i64> {
    let s = ids.len();
    let (mut pos, mut cur, mut i, mut g) = (vec![0i64; 3 * s], 0i64, 0usize, 0usize);
    while i < s {
        if ids[i] == image_pad {
            let (lh, lw) = (grids[g].0 / MERGE, grids[g].1 / MERGE); g += 1;
            for r in 0..lh { for c in 0..lw {
                let k = i + r * lw + c;
                pos[k] = cur * time_scale;          // temporel
                pos[s + k] = cur + r as i64;        // hauteur
                pos[2 * s + k] = cur + c as i64;    // largeur
            }}
            i += lh * lw; cur += lh.max(lw) as i64;
        } else {
            for d in 0..3 { pos[d * s + i] = cur; }
            cur += 1; i += 1;
        }
    }
    pos
}

//--------------------------------------------------------------------------------------------------

struct Past { name: String, f16: bool }

pub struct QwenOnnx {
    visual: Session, embed: Session, decoder: Session,
    tok: Tokenizer,
    letter_ids: HashMap<char, Vec<u32>>,
    pad_id: i64,
    past: Vec<Past>,
    kv_heads: usize, head_dim: usize, time_scale: i64,
    pub min_pixels: i32, pub max_pixels: i32,
}

//--------------------------------------------------------------------------------------------------

impl QwenOnnx {
    /// kv_heads / head_dim : config.json du modèle (3B : num_key_value_heads=2, hidden 2048 / 16 têtes = 128).
    /// tokens_per_second : config.vision_config.tokens_per_second.
    pub fn new(visual_path: &str, embed_path: &str, decoder_path: &str, tokenizer_json: &str,
               kv_heads: usize, head_dim: usize, tokens_per_second: i64, device: &str) -> anyhow::Result<Self> {
        let tok = Tokenizer::from_file(tokenizer_json).map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut letter_ids = HashMap::new();
        for l in LETTERS.chars().chain(std::iter::once(NONE_LETTER)) {
            // BPE d'octets de Qwen : l'espace est "Ġ" ; on accepte "A" et " A"
            let ids: Vec<u32> = [l.to_string(), format!("Ġ{l}")].iter().filter_map(|t| tok.token_to_id(t)).collect();
            letter_ids.insert(l, ids);
        }
        let pad_id = tok.token_to_id("<|image_pad|>").ok_or_else(|| anyhow::anyhow!("<|image_pad|> absent"))? as i64;
        let visual = setup_model_avec_cache(&visual_path, device)?;
        let embed = setup_model_avec_cache(&embed_path, device)?;
        let decoder = setup_model_avec_cache(&decoder_path, device)?;
        // [ORT] entrées du décodeur dont le nom commence par "past_key_values" + leur type (fp16 ou fp32)
        let past = decoder.inputs().iter().filter(|i| i.name().starts_with("past_key_values"))
            .map(|i| Past { name: i.name().to_string(), f16: format!("{:?}", i).contains("Float16") })
            .collect::<Vec<_>>();
        anyhow::ensure!(!past.is_empty(), "aucune entrée past_key_values : ce n'est pas le décodeur 'merged' attendu");
        Ok(Self { visual, embed, decoder, tok, letter_ids, pad_id, past, kv_heads, head_dim,
                  time_scale: tokens_per_second, min_pixels: 56 * 56, max_pixels: 28 * 28 * 768 })
    }
}

//--------------------------------------------------------------------------------------------------

impl VlmBackend for QwenOnnx {
    fn letter_ids(&self) -> &HashMap<char, Vec<u32>> { &self.letter_ids }

    fn first_token_logits(&mut self, prompt: &Prompt) -> anyhow::Result<Vec<f32>> {
        // 1) images -> patches ; texte = chat template Qwen2.5-VL (les morceaux sont concaténés SANS séparateur,
        //    comme le template HF ; une image devient <|vision_start|><|image_pad|>*n<|vision_end|>)
        let mut patches: Vec<f32> = Vec::new();
        let mut thw: Vec<i64> = Vec::new();
        let mut grids = Vec::new();
        let mut text = String::from("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n");
        for part in &prompt.parts {
            match part {
                Part::Text(t) => text.push_str(t),
                Part::Image(i) => {
                    let (p, gh, gw) = patchify(prompt.images[*i], self.min_pixels, self.max_pixels)?;
                    patches.extend(p); thw.extend([1, gh as i64, gw as i64]); grids.push((gh, gw));
                    text.push_str("<|vision_start|>");
                    for _ in 0..(gh / MERGE) * (gw / MERGE) { text.push_str("<|image_pad|>"); }
                    text.push_str("<|vision_end|>");
                }
            }
        }
        text.push_str("<|im_end|>\n<|im_start|>assistant\n");
        let ids: Vec<i64> = self.tok.encode(text, false).map_err(|e| anyhow::anyhow!("{e}"))?
            .get_ids().iter().map(|&i| i as i64).collect();
        let s = ids.len();

        // 2) encodeur visuel (toutes les images d'un coup ; vérifie que l'export accepte plusieurs images)
        let (n_img, n_patch) = (thw.len() / 3, patches.len() / 1176);
        let vis = self.visual.run(ort::inputs![
            "pixel_values" => Tensor::from_array(([n_patch, 1176usize], patches.into_boxed_slice()))?,
            "image_grid_thw" => Tensor::from_array(([n_img, 3usize], thw.into_boxed_slice()))?,
        ])?;
        let (vs, vd) = vis[0].try_extract_tensor::<f32>()?;           // [ORT] sortie par indice
        let hidden = vs[vs.len() - 1] as usize;
        let img_feats = vd.to_vec();
        drop(vis);

        // 3) embeddings des tokens + injection des features aux positions <|image_pad|>
        let emb = self.embed.run(ort::inputs!["input_ids" => Tensor::from_array(([1usize, s], ids.clone().into_boxed_slice()))?])?;
        let (_, ed) = emb[0].try_extract_tensor::<f32>()?;
        let mut embeds = ed.to_vec();
        drop(emb);
        anyhow::ensure!(embeds.len() == s * hidden, "dimension cachée incohérente");
        let mut k = 0usize;
        for (t, &id) in ids.iter().enumerate() {
            if id == self.pad_id {
                anyhow::ensure!((k + 1) * hidden <= img_feats.len(), "plus de <|image_pad|> que de features visuelles");
                embeds[t * hidden..(t + 1) * hidden].copy_from_slice(&img_feats[k * hidden..(k + 1) * hidden]);
                k += 1;
            }
        }
        anyhow::ensure!(k * hidden == img_feats.len(), "features visuelles non toutes placées ({k})");

        // 4) décodeur : cache vide, une passe, logits du dernier token
        let pos = rope_positions(&ids, self.pad_id, &grids, self.time_scale);
        let mut inputs: Vec<(String, ort::session::SessionInputValue<'_>)> = vec![   // [ORT] entrées dynamiques
            ("inputs_embeds".into(), Tensor::from_array(([1usize, s, hidden], embeds.into_boxed_slice()))?.into_dyn().into()),
            ("attention_mask".into(), Tensor::from_array(([1usize, s], vec![1i64; s].into_boxed_slice()))?.into_dyn().into()),
            ("position_ids".into(), Tensor::from_array(([3usize, 1, s], pos.into_boxed_slice()))?.into_dyn().into()),
        ];
        for p in &self.past {
            let shape = [1usize, self.kv_heads, 0, self.head_dim];
            let v = if p.f16 { Tensor::from_array((shape, Vec::<half::f16>::new().into_boxed_slice()))?.into_dyn() }
                    else      { Tensor::from_array((shape, Vec::<f32>::new().into_boxed_slice()))?.into_dyn() };
            inputs.push((p.name.clone(), v.into()));
        }
        let out = self.decoder.run(inputs)?;
        let (shape, lg) = out[0].try_extract_tensor::<f32>()?;        // logits fp32 attendus ; sinon utiliser la variante fp32
        let v = shape[shape.len() - 1] as usize;
        Ok(lg[lg.len() - v..].to_vec())                                // dernière position seulement
    }
}

//--------------------------------------------------------------------------------------------------
// END OF FILE
//--------------------------------------------------------------------------------------------------