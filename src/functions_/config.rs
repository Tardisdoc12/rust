//--------------------------------------------------------------------------------------------------
// Filename: config.rs
// Author: Jean Anquetil
// Date: Wednesday 18/11/2026
//--------------------------------------------------------------------------------------------------

use serde::Deserialize;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;

//--------------------------------------------------------------------------------------------------

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct Config {

    #[serde(default = "device_par_defaut")]
    device: &str,

    // Seuil de similarité pour accepter ou non une correspondance
    accept_sim: f32,
    accept_margin: f32,
    reject_sim: f32,

    // Marge pour les candidats
    cand_margin: Option<f32>,
    
    // Les nombres par défaut pour les photos et les candidats pour le VLM
    // => impacte le temps de traitement direct du VLM

    #[serde(default = "nbr_photo_par_defaut")]
    number_photos_candidates_for_vlm: usize,
    #[serde(default = "nbr_cdt_par_defaut")]
    number_candidate: usize,

    // paths_of_all_models:

    // Chemin vers le modèle YOLO 26
    yolo_26_path: &str,

    // Chemin vers le modèle Florence
    florence_path: &str,
    
    // Les différents fichiers de modèles pour SAM2
    sam2_path_encoder: &str,
    sam2_path_decoder: &str,
    
    // Embedder => C'est le modèle pour extraire les embeddings des objets détectés
    embedder_path: &str,
    
    // Chemin vers le dossier contenant la base d'identifiants pour l'embedder
    identifier_base_path: &str,
    
    // QwenOnnx => Modèle pour le traitement des produits difficiles à associer
    visual_path: &str, embed_path: &str, decoder_path: &str, tokenizer_json: &str,
    #[serde(default = "kv_heads_par_defaut")]
    kv_heads: usize,
    #[serde(default = "head_dim_par_defaut")] 
    head_dim: usize,
    #[serde(default = "tokens_per_second_par_defaut")]
    tokens_per_second: i64,

    // perms et early_conf pour le VLM
    perms: Option<usize>,
    early_conf: Option<f32>,
}

//--------------------------------------------------------------------------------------------------

pub fn lire_config(chemin: &str) -> Result<Config, Box<dyn Error>> {
    // 1. Ouvrir le fichier
    let fichier = File::open(chemin)?;
    // 2. Créer un lecteur tampon (buffer) pour optimiser la lecture
    let lecteur = BufReader::new(fichier);
    // 3. Désérialiser directement depuis le lecteur
    let config: Config = serde_json::from_reader(lecteur)?;
    
    Ok(config)
}

//--------------------------------------------------------------------------------------------------

// Valeurs par défaut pour les différents paramètres de configuration

fn device_par_defaut() -> &'static str {
    "cuda"
}

fn kv_heads_par_defaut() -> usize {
    2
}

fn head_dim_par_defaut() -> usize {
    128
}

fn nbr_photo_par_defaut() -> usize {
    2
}

fn nbr_cdt_par_defaut() -> usize {
    5
}

fn tokens_per_second_par_defaut() -> i64 {
    2
}

//--------------------------------------------------------------------------------------------------
// END OF FILE
//--------------------------------------------------------------------------------------------------