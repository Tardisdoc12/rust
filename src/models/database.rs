//--------------------------------------------------------------------------------------------------
// Filename: database.rs
// Author: Jean Anquetil
// Date: Wednesday 18/11/2026
//--------------------------------------------------------------------------------------------------

use std::{collections::HashMap, fs, path::Path};
use serde::Deserialize;

//--------------------------------------------------------------------------------------------------

#[derive(Deserialize)]
struct Item { ean: serde_json::Value, #[serde(default)] path: Option<String> }
#[derive(Deserialize)]
struct Meta {
    items: Vec<Item>,
    #[serde(default)] names: HashMap<String, String>,
    #[serde(default)] model: String,
    #[serde(default)] pretrained: String,
}

//--------------------------------------------------------------------------------------------------

pub struct Db {
    pub emb: Vec<f32>,          // m * dim, ligne par ligne (déjà normalisées)
    pub dim: usize,
    pub m: usize,
    pub eans: Vec<String>,      // EAN uniques triés (comme sorted(set(eans)))
    pub codes: Vec<usize>,      // code EAN de chaque item (longueur m)
    pub idx_by_code: Vec<Vec<usize>>,
    pub code_of: HashMap<String, usize>,
    pub paths: Vec<String>,     // chemin de la photo de chaque item (meta.json items[].path)
    pub names: HashMap<String, String>,
    pub model: String,
    pub pretrained: String,
}

//--------------------------------------------------------------------------------------------------

fn ean_str(v: &serde_json::Value) -> String {
    match v { serde_json::Value::String(s) => s.clone(), o => o.to_string() }
}

//--------------------------------------------------------------------------------------------------

/// Lecteur .npy minimal : version 1/2/3, little-endian, C-order, 2-D, dtype <f4 ou <f2.
fn read_npy(path: &Path) -> anyhow::Result<(Vec<f32>, usize, usize)> {
    let b = fs::read(path)?;
    anyhow::ensure!(b.len() > 10 && &b[..6] == b"\x93NUMPY", "pas un fichier .npy");
    let (hlen, off) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) }
                      else { (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12) };
    let header = std::str::from_utf8(&b[off..off + hlen])?;
    anyhow::ensure!(header.contains("'fortran_order': False"), "fortran_order non géré");
    let descr = header.split("'descr':").nth(1).and_then(|s| s.split('\'').nth(1))
        .ok_or_else(|| anyhow::anyhow!("descr introuvable"))?;
    let shape_s = header.split("'shape':").nth(1).and_then(|s| s.split('(').nth(1)).and_then(|s| s.split(')').next())
        .ok_or_else(|| anyhow::anyhow!("shape introuvable"))?;
    let dims: Vec<usize> = shape_s.split(',').filter_map(|t| t.trim().parse().ok()).collect();
    anyhow::ensure!(dims.len() == 2, "attendu 2 dimensions, reçu {:?}", dims);
    let data = &b[off + hlen..];
    let n = dims[0] * dims[1];
    let v: Vec<f32> = match descr {
        "<f4" => { anyhow::ensure!(data.len() >= n * 4, "fichier tronqué");
                   data.chunks_exact(4).take(n).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() }
        "<f2" => { anyhow::ensure!(data.len() >= n * 2, "fichier tronqué");
                   data.chunks_exact(2).take(n).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect() }
        d => anyhow::bail!("dtype non géré: {d}"),
    };
    Ok((v, dims[0], dims[1]))
}

//--------------------------------------------------------------------------------------------------

impl Db {
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        let meta: Meta = serde_json::from_slice(&fs::read(path.join("meta.json"))?)?;
        let (emb, m, dim) = read_npy(&path.join("embeddings.npy"))?;
        anyhow::ensure!(m == meta.items.len(), "embeddings.npy ({m}) et meta.json ({}) ne correspondent pas", meta.items.len());
        let item_eans: Vec<String> = meta.items.iter().map(|i| ean_str(&i.ean)).collect();
        let mut eans = item_eans.clone(); eans.sort(); eans.dedup();
        let code_of: HashMap<String, usize> = eans.iter().cloned().enumerate().map(|(k, e)| (e, k)).collect();
        let codes: Vec<usize> = item_eans.iter().map(|e| code_of[e]).collect();
        let mut idx_by_code = vec![Vec::new(); eans.len()];
        for (i, &c) in codes.iter().enumerate() { idx_by_code[c].push(i); }
        let paths: Vec<String> = meta.items.iter().map(|i| i.path.clone().unwrap_or_default()).collect();
        Ok(Self { emb, dim, m, eans, paths, codes, idx_by_code, code_of, names: meta.names, model: meta.model, pretrained: meta.pretrained })
    }

    /// Photos de référence du produit `c` les plus proches de la requête, sans quasi-doublons (cos < dedup).
    /// `sims` : similarités de tous les items (1er résultat de rank).
    pub fn pick_refs(&self, sims: &[f32], c: usize, n: usize, dedup: f32) -> Vec<&str> {
        let mut idx = self.idx_by_code[c].clone();
        idx.sort_by(|&a, &b| sims[b].partial_cmp(&sims[a]).unwrap_or(std::cmp::Ordering::Equal));
        let row = |i: usize| &self.emb[i * self.dim..(i + 1) * self.dim];
        let mut chosen: Vec<usize> = Vec::new();
        for i in idx {
            if chosen.iter().all(|&j| row(i).iter().zip(row(j)).map(|(a, b)| a * b).sum::<f32>() < dedup) { chosen.push(i); }
            if chosen.len() >= n { break; }
        }
        chosen.into_iter().map(|i| self.paths[i].as_str()).collect()
    }

    pub fn name(&self, c: usize) -> &str {
        self.names.get(&self.eans[c]).map(|s| s.as_str()).unwrap_or("")
    }

    /// Similarité max par code EAN (sans tri) : ce que rank calcule d'abord, en moins cher.
    pub fn best(&self, q: &[f32]) -> Vec<f32> {
        assert_eq!(q.len(), self.dim, "dimension de la requête");
        let mut best = vec![f32::NEG_INFINITY; self.eans.len()];
        for (i, row) in self.emb.chunks_exact(self.dim).enumerate() {
            let s: f32 = row.iter().zip(q).map(|(a, b)| a * b).sum();
            let c = self.codes[i]; if s > best[c] { best[c] = s; }
        }
        best
    }

    /// Équivalent de DB.rank : (sims de tous les items, codes EAN triés par sim décroissante, sim max par EAN).
    /// `q` : embedding normalisé de la requête (1152 floats).
    pub fn rank(&self, q: &[f32]) -> (Vec<f32>, Vec<usize>, Vec<f32>) {
        assert_eq!(q.len(), self.dim, "dimension de la requête");
        let sims: Vec<f32> = self.emb.chunks_exact(self.dim)
            .map(|row| row.iter().zip(q).map(|(a, b)| a * b).sum()).collect();
        let mut best = vec![f32::NEG_INFINITY; self.eans.len()];
        for (i, &s) in sims.iter().enumerate() {
            let c = self.codes[i]; if s > best[c] { best[c] = s; }
        }
        let mut order: Vec<usize> = (0..self.eans.len()).collect();
        order.sort_by(|&a, &b| best[b].partial_cmp(&best[a]).unwrap_or(std::cmp::Ordering::Equal));
        (sims, order, best)
    }

    /// Top-k candidats (EAN, nom, similarité) — ce que tu passeras ensuite au VLM comme options.
    pub fn top_k(&self, q: &[f32], k: usize) -> Vec<(String, String, f32)> {
        let (_, order, best) = self.rank(q);
        order.into_iter().take(k).map(|c| (self.eans[c].clone(), self.name(c).to_string(), best[c])).collect()
    }
}

//--------------------------------------------------------------------------------------------------
// END OF FILE
//--------------------------------------------------------------------------------------------------