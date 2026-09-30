//--------------------------------------------------------------------------------------------------
// Filename: label_resolver.rs
// Author: Jean Anquetil
// Date: 2026-09-30
//--------------------------------------------------------------------------------------------------

use std::collections::HashMap;

//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
struct LabelInfo {
    sizes: Vec<(String, f64, String)>,
    master_product_id: Option<String>,
}


/// Connexion paresseuse + cache, durée de vie = un appel à `process`.
struct LabelResolver {
    conn: Option<ConnecteurServer>,
    connect_failed: bool,
    // label -> liste (ean, hauteur) ; None = famille inconnue
    cache: HashMap<String, Option<Vec<(String, f64, String)>>>,
}

//--------------------------------------------------------------------------------------------------

impl LabelResolver {
    fn new() -> Self {
        Self { conn: None, connect_failed: false, cache: HashMap::new() }
    }

    fn conn(&mut self) -> Option<&mut ConnecteurServer> {
        if self.conn.is_none() && !self.connect_failed {
            match ConnecteurServer::connect() {
                Ok(c) => self.conn = Some(c),
                Err(e) => {
                    eprintln!("Erreur de connexion à la base : {e}");
                    self.connect_failed = true; // on n'insiste pas pour les produits suivants
                }
            }
        }
        self.conn.as_mut()
    }

    fn sizes_for(&mut self, label: &str) -> Option<&Vec<(String, f64, String)>> {
        if !self.cache.contains_key(label) {
            let fetched = self.conn().and_then(|c| {
                let family_id = c.get_family_id(label).ok().flatten()?;
                let list = c.get_size_of_products(family_id).ok()?;
                if list.is_empty() {
                    None
                } else {
                    Some(LabelInfo {
                        sizes: list.into_iter().map(|(id, ean, h, _l)| (ean, h, id)).collect(),
                        master_product_id: None,
                    })
                }
            });
            self.cache.insert(label.to_string(), fetched);
        }
        self.cache.get(label).and_then(|o| o.as_ref())
    }

    fn resolve(&mut self, detection: &mut Detection) {
        if detection.label.is_empty()
            || detection.label == "OOD"
            || detection.height_cm <= 0.0
        {
            return;
        }

        let height = detection.height_cm as f64;
        let Some(sizes) = self.sizes_for(&detection.label) else { return };

        if let Some((ean, _, id)) = sizes
            .iter()
            .min_by(|a, b| (height - a.1).abs().total_cmp(&(height - b.1).abs()))
        {
            detection.label = ean.clone();
            detection.master_product_id = id.clone();
        }
    }

    fn master_product_id_for(&mut self, label: &str) -> &Vec<(String, f64, String)> {
        if !self.cache.contains_key(label) {
            let fetched = self.conn().and_then(|c| {
                let master_product_id = c.get_master_product(label).ok().flatten()?;
                if master_product_id.is_empty() {
                    None
                } else {
                    Some((label.to_string(), 0, master_product_id))
                }
            });
            self.cache.insert(label.to_string(), fetched);
        }
        self.cache.get(label).and_then(|o| o.as_ref())
    }

    fn get_master_product_id(&mut self, label: &str) -> Option<String> {
        self.master_product_id_for(label)
            .iter()
            .find_map(|(_, _, master_product_id)| Some(master_product_id.clone()))
    }
}

//--------------------------------------------------------------------------------------------------
// End of file
//--------------------------------------------------------------------------------------------------