//--------------------------------------------------------------------------------------------------
// Filename: connecteur.rs
// Port de connecteur.py
//--------------------------------------------------------------------------------------------------

use postgres::{Client, NoTls};

//--------------------------------------------------------------------------------------------------
// Requêtes (Queries.py)
//--------------------------------------------------------------------------------------------------

const GET_FAMILY_ID: &str = r#"
    SELECT "family_id" FROM "default$default"."MasterProduct"
    WHERE "Ean" = $1
"#;

//--------------------------------------------------------------------------------------------------

const GET_SIZE_OF_PRODUCTS: &str = r#"
    SELECT "id"::text,
           "Ean",
           CASE WHEN replace(btrim("Hauteur_du_produit"), ',', '.') ~ '^[0-9]+([.][0-9]+)?$'
                THEN replace(btrim("Hauteur_du_produit"), ',', '.')::float8 END,
           CASE WHEN replace(btrim("Largeur_du_produit"), ',', '.') ~ '^[0-9]+([.][0-9]+)?$'
                THEN replace(btrim("Largeur_du_produit"), ',', '.')::float8 END
    FROM "default$default"."MasterProduct"
    WHERE "family_id" = $1
"#;

//--------------------------------------------------------------------------------------------------

const GET_MASTER_PRODUCT: &str = r#"
    SELECT "id"::text FROM "default$default"."MasterProduct"
    WHERE "Ean" = $1
"#;

//--------------------------------------------------------------------------------------------------

pub struct ConnecteurServer {
    client: Client,
}

impl ConnecteurServer {
    /// Équivalent de __init__ + __enter__ : lit les identifiants dans les
    /// variables d'environnement et ouvre directement la connexion.
    pub fn connect() -> anyhow::Result<Self> {
        let identifiant = std::env::var("IDENTIFIANT")?;
        let password_db = std::env::var("PASSWORDDB")?;
        let database = std::env::var("DATABASE")?;
        let hostname = std::env::var("DB_HOST")?;
        let port: u16 = std::env::var("PORT")
            .unwrap_or_else(|_| "5432".to_string())
            .parse()?;

        let conn_str = format!(
            "host={} port={} dbname={} user={} password={} connect_timeout=5",
            hostname, port, database, identifiant, password_db
        );

        let client = Client::connect(&conn_str, NoTls)?;

        Ok(Self { client })
    }

    /// Équivalent de get_family_id
    pub fn get_family_id(&mut self, ref_master_product: &str) -> anyhow::Result<Option<i32>> {
        let row = self.client.query_opt(GET_FAMILY_ID, &[&ref_master_product])?;
        let id: Option<i32> = match row {
            Some(r) => r.try_get(0)?,
            None => None,
        };
        Ok(id.filter(|&f| f != 0)) // 0 = produit sans famille
    }

    pub fn get_master_product(&mut self, ref_ean_master_product: &str) -> anyhow::Result<Option<String>> {
        let row = self.client.query_opt(GET_MASTER_PRODUCT, &[&ref_ean_master_product])?;
        match row {
            Some(r) => Ok(r.try_get(0)?),
            None => Ok(None),
        }
    }

    /// Équivalent de get_size_of_products
    /// Retourne (Ean, Hauteur_du_produit, Largeur_du_produit) pour chaque ligne
    pub fn get_size_of_products(
        &mut self,
        family_id: i32,
    ) -> anyhow::Result<Vec<(String, String, f64, f64)>> {
        let rows = self.client.query(GET_SIZE_OF_PRODUCTS, &[&family_id])?;

        let mut out = Vec::with_capacity(rows.len());
        for r in rows {
            let id: String = r.try_get(0)?;
            let ean: String = r.try_get(1)?;
            let h: Option<f64> = r.try_get(2)?;
            let l: Option<f64> = r.try_get(3)?;
            match h {
                Some(h) if h > 0.0 => out.push((id, ean, h, l.unwrap_or(0.0))),
                _ => eprintln!("[db] hauteur absente ou nulle pour l'EAN {ean}, ignorée"),
            }
        }
        Ok(out)
    }
}

//--------------------------------------------------------------------------------------------------
// End of file
//--------------------------------------------------------------------------------------------------