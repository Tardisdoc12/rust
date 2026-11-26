//--------------------------------------------------------------------------------------------------
// Filename: qwen_onnx.rs
// Author: Jean Anquetil
// Date: Wednesday 18/11/2026
//--------------------------------------------------------------------------------------------------

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::time::Instant;
use opencv::{core::{Mat, Size, CV_32F}, imgproc, prelude::*};
use ort::{session::{RunOptions, Session}, value::Tensor};
use tokenizers::Tokenizer;

use crate::models::vlm::{Part, Prompt, VlmBackend, LETTERS, NONE_LETTER};
use crate::functions_::setup_model_avec_cache::setup_model_avec_cache;

//--------------------------------------------------------------------------------------------------

const FACTOR: i32 = 28;           // patch 14 x merge 2
const P: usize = 14;
const MERGE: usize = 2;
const PATCH_DIM: usize = 3 * 2 * P * P;   // 1176 = canaux x temporel x 14 x 14
/// Nombre max d'images dont on garde les features visuelles en cache (~2 Mo chacune à 256 tokens x 2048).
const FEAT_CACHE_MAX: usize = 256;
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
    let px: Vec<f32> = f
        .data_typed::<opencv::core::Vec3f>()?   // lit des triplets (CV_32FC3)
        .iter()
        .flat_map(|v| v.0)                       // -> [c0, c1, c2, c0, c1, c2, ...] (HWC)
        .collect();
    let (h, w) = (h as usize, w as usize);
    let (gh, gw) = (h / P, w / P);
    let (a_n, b_n) = (gh / MERGE, gw / MERGE);
    let mut out = vec![0f32; gh * gw * PATCH_DIM];
    for a in 0..a_n { for b in 0..b_n { for mh in 0..MERGE { for mw in 0..MERGE {
        let row = ((a * b_n + b) * MERGE + mh) * MERGE + mw;
        for c in 0..3 { for ph in 0..P { for pw in 0..P {
            let y = (a * MERGE + mh) * P + ph; let x = (b * MERGE + mw) * P + pw;
            let v = (px[(y * w + x) * 3 + c] - MEAN[c]) / STD[c];
            for t in 0..2 {                                 // image fixe : les 2 images temporelles sont identiques
                out[row * PATCH_DIM + ((c * 2 + t) * P + ph) * P + pw] = v;
            }
        }}}
    }}}}
    Ok((out, gh, gw))
}

//--------------------------------------------------------------------------------------------------

/// Positions (3,S) = port du get_rope_index du script officiel. grids = (gh, gw) AVANT fusion 2x2 de chaque image.
/// `time_scale` = tokens_per_second (config.vision_config.tokens_per_second) x second_per_grid (1 dans le script).
fn rope_positions(ids: &[i64], image_pad: i64, grids: &[(usize, usize)], _time_scale: i64) -> Vec<i64> {
    let s = ids.len();
    let (mut pos, mut cur, mut i, mut g) = (vec![0i64; 3 * s], 0i64, 0usize, 0usize);
    while i < s {
        if ids[i] == image_pad {
            let (lh, lw) = (grids[g].0 / MERGE, grids[g].1 / MERGE); g += 1;
            for r in 0..lh { for c in 0..lw {
                let k = i + r * lw + c;
                pos[k] = cur;                       // temporel : pour une IMAGE = position de départ, sans facteur
                                                    // (transformers : time_interval = 1 pour les images, fix #45325 ;
                                                    //  le script d'exemple onnx-community multipliait à tort par tokens_per_second)
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

/// Options de run qui rendent au GPU la mémoire inutilisée du pool (arena) à la fin du run.
/// Sans ça, chaque session garde à vie le pic de mémoire atteint (séquence la plus longue, etc.).
/// Créées à chaque appel : coût négligeable, et évite tout souci de Send/Sync sur la struct.
fn shrink_opts() -> anyhow::Result<RunOptions<ort::session::NoSelectedOutputs>> {
    let mut opts = RunOptions::new()?;
    opts.set("memory.enable_memory_arena_shrinkage", "gpu:0")?;
    Ok(opts)
}

/// Empreinte du contenu d'une image (taille + type + pixels). Deux Mat identiques pixel à pixel
/// ont la même clé, même si ce sont des objets différents (fit_max_side recrée la Mat à chaque appel).
fn image_key(img: &Mat) -> anyhow::Result<u64> {
    let owned;
    let m = if img.is_continuous() { img } else { owned = img.try_clone()?; &owned };
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (m.rows(), m.cols(), m.typ()).hash(&mut h);
    m.data_bytes()?.hash(&mut h);
    Ok(h.finish())
}

/// Grille de patches (gh, gw) qu'aura l'image après smart_resize, sans faire le patchify.
fn grid_of(img: &Mat, min_pixels: i32, max_pixels: i32) -> (usize, usize) {
    let (h, w) = smart_resize(img.rows(), img.cols(), min_pixels, max_pixels);
    (h as usize / P, w as usize / P)
}

/// true si l'entrée/sortie `name` de la session est en float16 (même détection que pour past_key_values).
fn is_f16(outlets: &[ort::value::Outlet], name: &str) -> bool {
    outlets.iter().find(|o| o.name() == name).map(|o| format!("{:?}", o).contains("Float16")).unwrap_or(false)
}

/// Vec<f32> -> tenseur ORT en f32 ou f16 selon ce qu'attend le modèle.
fn tensor_f32_or_f16<const N: usize>(shape: [usize; N], data: Vec<f32>, f16: bool) -> anyhow::Result<ort::value::DynValue> {
    Ok(if f16 {
        let h: Vec<half::f16> = data.into_iter().map(half::f16::from_f32).collect();
        Tensor::from_array((shape, h.into_boxed_slice()))?.into_dyn()
    } else {
        Tensor::from_array((shape, data.into_boxed_slice()))?.into_dyn()
    })
}

/// Extrait une sortie en (shape, Vec<f32>), qu'elle soit en f32 ou en f16 : le type est lu sur le tenseur
/// lui-même au moment de l'extraction (plus fiable que la détection faite au chargement).
fn extract_f32(v: &ort::value::DynValue) -> anyhow::Result<(Vec<i64>, Vec<f32>)> {
    if let Ok((sh, d)) = v.try_extract_tensor::<f32>() {
        return Ok((sh.to_vec(), d.to_vec()));
    }
    let (sh, d) = v.try_extract_tensor::<half::f16>()
        .map_err(|e| anyhow::anyhow!("sortie ni f32 ni f16 : {e}"))?;
    Ok((sh.to_vec(), d.iter().map(|x| x.to_f32()).collect()))
}

/// Logits du DERNIER token seulement, convertis en f32. La sortie complète fait [1, S, vocab]
/// (≈ 1850 x 152 000 valeurs) : la convertir entièrement coûtait ~800 ms pour n'en garder qu'une ligne.
fn last_logits(v: &ort::value::DynValue) -> anyhow::Result<Vec<f32>> {
    if let Ok((sh, d)) = v.try_extract_tensor::<f32>() {
        let n = sh[sh.len() - 1] as usize;
        return Ok(d[d.len() - n..].to_vec());
    }
    let (sh, d) = v.try_extract_tensor::<half::f16>()
        .map_err(|e| anyhow::anyhow!("logits ni f32 ni f16 : {e}"))?;
    let n = sh[sh.len() - 1] as usize;
    Ok(d[d.len() - n..].iter().map(|x| x.to_f32()).collect())
}

//--------------------------------------------------------------------------------------------------

struct Past { name: String, f16: bool }

/// Une image prête pour l'encodeur visuel.
struct ImagePatches { patches: Vec<f32>, gh: usize, gw: usize }

pub struct QwenOnnx {
    visual: Session, embed: Session, decoder: Session,
    tok: Tokenizer,
    letter_ids: HashMap<char, Vec<u32>>,
    pad_id: i64,
    past: Vec<Past>,
    kv_heads: usize, head_dim: usize, time_scale: i64,
    pub min_pixels: i32, pub max_pixels: i32,
    /// Cache des features visuelles par contenu d'image. Les photos de référence des candidats reviennent
    /// à chaque permutation et d'un produit à l'autre : on ne les encode qu'une fois. Résultat identique
    /// (l'encodeur traite chaque image indépendamment), seul le temps change.
    feat_cache: HashMap<u64, Vec<f32>>,
    /// Types détectés au chargement : permet d'utiliser indifféremment les variantes fp32 / fp16 / q4 / q4f16.
    vis_in_f16: bool, vis_out_f16: bool, dec_in_f16: bool, dec_out_f16: bool,
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
        // embed_tokens = simple table de correspondance (~1,2 Go en fp32) : aussi rapide sur CPU, libère la VRAM
        let embed = setup_model_avec_cache(&embed_path, "cpu")?;
        let decoder = setup_model_avec_cache(&decoder_path, device)?;
        // [ORT] entrées du décodeur dont le nom commence par "past_key_values" + leur type (fp16 ou fp32)
        let past = decoder.inputs().iter().filter(|i| i.name().starts_with("past_key_values"))
            .map(|i| Past { name: i.name().to_string(), f16: format!("{:?}", i).contains("Float16") })
            .collect::<Vec<_>>();
        anyhow::ensure!(!past.is_empty(), "aucune entrée past_key_values : ce n'est pas le décodeur 'merged' attendu");
        let vis_in_f16 = is_f16(visual.inputs(), "pixel_values");
        let vis_out_f16 = visual.outputs().first().map(|o| format!("{:?}", o).contains("Float16")).unwrap_or(false);
        let dec_in_f16 = is_f16(decoder.inputs(), "inputs_embeds");
        let dec_out_f16 = decoder.outputs().first().map(|o| format!("{:?}", o).contains("Float16")).unwrap_or(false);
        eprintln!("[vlm] types : visuel entrée {} / sortie {} | décodeur inputs_embeds {} / logits {}",
            if vis_in_f16 { "f16" } else { "f32" }, if vis_out_f16 { "f16" } else { "f32" },
            if dec_in_f16 { "f16" } else { "f32" }, if dec_out_f16 { "f16" } else { "f32" });

        Ok(Self { visual, embed, decoder, tok, letter_ids, pad_id, past, kv_heads, head_dim,
                  time_scale: tokens_per_second, min_pixels: 56 * 56, max_pixels: 28 * 28 * 256,   // 256 tokens max / image
                  feat_cache: HashMap::new(),
                  vis_in_f16, vis_out_f16, dec_in_f16, dec_out_f16 })
    }

    /// Encodeur visuel sur UNE image : retourne ses features ((gh/2)*(gw/2) x hidden, aplaties).
    /// L'export ONNX (onnx-community) ne gère qu'une image par appel : avec plusieurs images
    /// concaténées, le Reshape '/model/vision_rope/pad/hidden_5d' échoue.
    fn encode_image(&mut self, img: ImagePatches, hidden: usize) -> anyhow::Result<Vec<f32>> {
        let n_patch = img.gh * img.gw;
        anyhow::ensure!(img.patches.len() == n_patch * PATCH_DIM,
            "patches incohérents : {} valeurs pour {}x{} patches", img.patches.len(), img.gh, img.gw);

        let opts = shrink_opts()?;
        let vis = self.visual.run_with_options(ort::inputs![
            "pixel_values" => tensor_f32_or_f16([n_patch, PATCH_DIM], img.patches, self.vis_in_f16)?,
            "image_grid_thw" => Tensor::from_array(([1usize, 3usize], vec![1i64, img.gh as i64, img.gw as i64].into_boxed_slice()))?,
        ], &opts)?;
        let (vs, vd) = extract_f32(&vis[0])?;      // [ORT] sortie par indice, convertie en f32
        let out_hidden = vs[vs.len() - 1] as usize;
        anyhow::ensure!(out_hidden == hidden,
            "dimension cachée visuelle ({out_hidden}) différente de celle des embeddings texte ({hidden})");

        let expected_tokens = (img.gh / MERGE) * (img.gw / MERGE);
        anyhow::ensure!(vd.len() == expected_tokens * hidden,
            "encodeur visuel : {} valeurs, attendu {} tokens x {}", vd.len(), expected_tokens, hidden);
        Ok(vd)
    }
}

//--------------------------------------------------------------------------------------------------

impl VlmBackend for QwenOnnx {
    fn letter_ids(&self) -> &HashMap<char, Vec<u32>> { &self.letter_ids }

    fn first_token_logits(&mut self, prompt: &Prompt) -> anyhow::Result<Vec<f32>> {
        let t_total = Instant::now();
        // 1) texte = chat template Qwen2.5-VL (les morceaux sont concaténés SANS séparateur, comme le template HF ;
        //    une image devient <|vision_start|><|image_pad|>*n<|vision_end|>). La grille est calculée sans patchify.
        let mut order: Vec<usize> = Vec::new();          // indices dans prompt.images, dans l'ordre des <|image_pad|>
        let mut grids = Vec::new();
        let mut text = String::from("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n");
        for part in &prompt.parts {
            match part {
                Part::Text(t) => text.push_str(t),
                Part::Image(i) => {
                    let (gh, gw) = grid_of(prompt.images[*i], self.min_pixels, self.max_pixels);
                    grids.push((gh, gw));
                    order.push(*i);
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

        // 2) embeddings des tokens (donne aussi la dimension cachée, même sans image)
        let t_embed = Instant::now();
        let emb = self.embed.run(ort::inputs!["input_ids" => Tensor::from_array(([1usize, s], ids.clone().into_boxed_slice()))?])?;
        let (es, mut embeds) = extract_f32(&emb[0])?;                  // f32 ou f16 selon la variante d'embed_tokens
        let hidden = es[es.len() - 1] as usize;
        drop(emb);
        anyhow::ensure!(embeds.len() == s * hidden, "dimension cachée incohérente");

        let d_embed = t_embed.elapsed();

        // 3) encodeur visuel : UNE image par appel, et seulement si elle n'est pas déjà en cache
        let t_visual = Instant::now();
        let (mut hits, mut misses) = (0usize, 0usize);
        let mut img_feats: Vec<f32> = Vec::new();
        for (&i, &(gh, gw)) in order.iter().zip(&grids) {
            let img = prompt.images[i];
            let key = image_key(img)?;
            if let Some(f) = self.feat_cache.get(&key) {
                img_feats.extend_from_slice(f);
                hits += 1;
                continue;
            }
            misses += 1;
            let (p, pgh, pgw) = patchify(img, self.min_pixels, self.max_pixels)?;
            anyhow::ensure!((pgh, pgw) == (gh, gw), "grille incohérente entre grid_of et patchify");
            let feats = self.encode_image(ImagePatches { patches: p, gh, gw }, hidden)?;
            img_feats.extend_from_slice(&feats);
            if self.feat_cache.len() >= FEAT_CACHE_MAX { self.feat_cache.clear(); }   // purge simple
            self.feat_cache.insert(key, feats);
        }
        let d_visual = t_visual.elapsed();

        // 4) injection des features aux positions <|image_pad|>
        let mut k = 0usize;
        for (t, &id) in ids.iter().enumerate() {
            if id == self.pad_id {
                anyhow::ensure!((k + 1) * hidden <= img_feats.len(), "plus de <|image_pad|> que de features visuelles");
                embeds[t * hidden..(t + 1) * hidden].copy_from_slice(&img_feats[k * hidden..(k + 1) * hidden]);
                k += 1;
            }
        }
        anyhow::ensure!(k * hidden == img_feats.len(), "features visuelles non toutes placées ({k})");

        // 5) décodeur : cache vide, une passe, logits du dernier token
        let pos = rope_positions(&ids, self.pad_id, &grids, self.time_scale);
        let mut inputs: Vec<(String, ort::session::SessionInputValue<'_>)> = vec![   // [ORT] entrées dynamiques
            ("inputs_embeds".into(), tensor_f32_or_f16([1usize, s, hidden], embeds, self.dec_in_f16)?.into()),
            ("attention_mask".into(), Tensor::from_array(([1usize, s], vec![1i64; s].into_boxed_slice()))?.into_dyn().into()),
            ("position_ids".into(), Tensor::from_array(([3usize, 1, s], pos.into_boxed_slice()))?.into_dyn().into()),
        ];
        for p in &self.past {
            let shape = [1usize, self.kv_heads, 0, self.head_dim];
            let v = if p.f16 { Tensor::from_array((shape, Vec::<half::f16>::new().into_boxed_slice()))?.into_dyn() }
                    else      { Tensor::from_array((shape, Vec::<f32>::new().into_boxed_slice()))?.into_dyn() };
            inputs.push((p.name.clone(), v.into()));
        }
        let t_decoder = Instant::now();
        let opts = shrink_opts()?;
        let out = self.decoder.run_with_options(inputs, &opts)?;
        let d_decoder = t_decoder.elapsed();
        let logits = last_logits(&out[0])?;                             // dernière position seulement

        eprintln!(
            "[vlm timing] total {:>6.0} ms | visuel {:>6.0} ms ({} images : {} encodées, {} en cache) | embed {:>4.0} ms | décodeur {:>6.0} ms ({} tokens)",
            t_total.elapsed().as_secs_f64() * 1e3,
            d_visual.as_secs_f64() * 1e3, hits + misses, misses, hits,
            d_embed.as_secs_f64() * 1e3,
            d_decoder.as_secs_f64() * 1e3, s,
        );
        Ok(logits)
    }
}

//--------------------------------------------------------------------------------------------------
// END OF FILE
//--------------------------------------------------------------------------------------------------