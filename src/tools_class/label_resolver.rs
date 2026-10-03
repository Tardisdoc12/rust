//--------------------------------------------------------------------------------------------------
// Filename: label_resolver.rs
// Author: Jean Anquetil
// Date: 2026-10-03
//--------------------------------------------------------------------------------------------------

use std::collections::HashMap;
use crate::tools_class::connecteur::ConnecteurServer;
use crate::detections::detections::Detection;

//--------------------------------------------------------------------------------------------------

pub struct LabelResolver {
    master_id: HashMap<String, String>,              // EAN -> id
    ean_family: HashMap<String, i32>,                // EAN -> famille
    family_sizes: HashMap<i32, Vec<(String, f64)>>,  // famille -> [(EAN, hauteur)]
}

//--------------------------------------------------------------------------------------------------

impl LabelResolver {
    pub fn load() -> anyhow::Result<Self> {
        let rows = ConnecteurServer::connect()?.get_catalog()?; // connexion fermée ici

        let mut master_id = HashMap::new();
        let mut ean_family = HashMap::new();
        let mut family_sizes: HashMap<i32, Vec<(String, f64)>> = HashMap::new();

        for r in rows {
            if let Some(f) = r.family_id.filter(|&f| f != 0) {
                ean_family.insert(r.ean.clone(), f);
                match r.height {
                    Some(h) if h > 0.0 => family_sizes.entry(f).or_default().push((r.ean.clone(), h)),
                    _ => eprintln!("[db] hauteur absente ou nulle pour l'EAN {}, ignorée", r.ean),
                }
            }
            master_id.insert(r.ean, r.id);
        }
        Ok(Self { master_id, ean_family, family_sizes })
    }

    pub fn get_master_product_id(&self, ean: &str) -> Option<String> {
        self.master_id.get(ean).cloned()
    }

    pub fn resolve(&self, detection: &mut Detection) {
        if detection.label.is_empty(){
            return;
        }

        // Affinage par la taille seulement si on a une homographie
        if detection.height_cm > 0.0  && detection.label != "OOB" && detection.label != "OOD" {
            let height = detection.height_cm as f64;
            let best = self
                .ean_family
                .get(&detection.label)
                .and_then(|f| self.family_sizes.get(f))
                .and_then(|sizes| {
                    sizes.iter()
                        .min_by(|a, b| (height - a.1).abs().total_cmp(&(height - b.1).abs()))
                        .map(|(ean, _)| ean.clone())
                });
            if let Some(ean) = best {
                detection.label = ean;
            }
        }

        match self.get_master_product_id(&detection.label) {
            Some(id) => detection.master_product_id = id,
            None => eprintln!("[resolver] pas de master product pour l'EAN {}", detection.label),
        }
    }
}

//--------------------------------------------------------------------------------------------------
// End of file
//--------------------------------------------------------------------------------------------------