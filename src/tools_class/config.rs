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
pub struct Config {

    #[serde(default = "device_par_defaut")]
    pub device: String,

    // Seuil de similarité pour accepter ou non une correspondance
    pub accept_sim: f32,
    pub accept_margin: f32,
    pub reject_sim: f32,

    // Marge pour les candidats
    pub cand_margin: Option<f32>,
    
    // Les nombres par défaut pour les photos et les candidats pour le VLM
    // => impacte le temps de traitement direct du VLM

    #[serde(default = "nbr_photo_par_defaut")]
    pub number_photos_candidates_for_vlm: usize,
    #[serde(default = "nbr_cdt_par_defaut")]
    pub number_candidate: usize,

    // paths_of_all_models:

    // Chemin vers le modèle YOLO 26
    pub yolo_26_path: String,

    // Chemin vers le modèle Florence
    pub florence_path: String,
    
    // Les différents fichiers de modèles pour SAM2
    pub sam2_path_encoder: String,
    pub sam2_path_decoder: String,
    
    // Embedder => C'est le modèle pour extraire les embeddings des objets détectés
    pub embedder_path: String,
    
    // Chemin vers le dossier contenant la base d'identifiants pour l'embedder
    pub identifier_base_path: String,
    
    // QwenOnnx => Modèle pour le traitement des produits difficiles à associer
    pub visual_path: String, pub embed_path: String, pub decoder_path: String, pub tokenizer_json: String,
    #[serde(default = "kv_heads_par_defaut")]
    pub kv_heads: usize,
    #[serde(default = "head_dim_par_defaut")] 
    pub head_dim: usize,
    #[serde(default = "tokens_per_second_par_defaut")]
    pub tokens_per_second: i64,

    // perms et early_conf pour le VLM
    #[serde(default = "perms_par_defaut")]
    pub perms: usize,
    #[serde(default = "early_conf_par_defaut")]
    pub early_conf: f32,
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

fn device_par_defaut() -> String {
    "cuda".to_string()
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

fn perms_par_defaut() -> usize {
    2
}

fn early_conf_par_defaut() -> f32 {
    0.9
}

//--------------------------------------------------------------------------------------------------
// END OF FILE
//--------------------------------------------------------------------------------------------------