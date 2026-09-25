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

const GET_SIZE_OF_PRODUCTS: &str = r#"
    SELECT "Ean", "Hauteur_du_produit", "Largeur_du_produit"
    FROM "default$default"."MasterProduct"
    WHERE "family_id" = $1
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
        let hostname = std::env::var("HOSTNAME")?;
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
        let row = self
            .client
            .query_opt(GET_FAMILY_ID, &[&ref_master_product])?;

        Ok(row.map(|r| r.get::<_, i32>(0)))
    }

    /// Équivalent de get_size_of_products
    /// Retourne (Ean, Hauteur_du_produit, Largeur_du_produit) pour chaque ligne
    pub fn get_size_of_products(
        &mut self,
        family_id: i32,
    ) -> anyhow::Result<Vec<(String, f64, f64)>> {
        let rows = self.client.query(GET_SIZE_OF_PRODUCTS, &[&family_id])?;

        Ok(rows
            .into_iter()
            .map(|r| (r.get::<_, String>(0), r.get::<_, f64>(1), r.get::<_, f64>(2)))
            .collect())
    }
}

// Pas de __exit__ à porter : Client implémente déjà Drop, la connexion
// se ferme automatiquement quand ConnecteurServer sort de scope.

//--------------------------------------------------------------------------------------------------
// End of file
//--------------------------------------------------------------------------------------------------