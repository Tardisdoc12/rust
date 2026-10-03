//--------------------------------------------------------------------------------------------------
// Filename: connecteur.rs
// Author: Jean Anquetil
// Date: 2026-10-03
//--------------------------------------------------------------------------------------------------

use postgres::{Client, NoTls};

const GET_CATALOG: &str = r#"
    SELECT "id"::text,
           "Ean",
           "family_id",
           CASE WHEN replace(btrim("Hauteur_du_produit"), ',', '.') ~ '^[0-9]+([.][0-9]+)?$'
                THEN replace(btrim("Hauteur_du_produit"), ',', '.')::float8 END
    FROM "default$default"."MasterProduct"
"#;

//--------------------------------------------------------------------------------------------------

pub struct CatalogRow {
    pub id: String,
    pub ean: String,
    pub family_id: Option<i32>,
    pub height: Option<f64>,
}

//--------------------------------------------------------------------------------------------------

pub struct ConnecteurServer {
    client: Client,
}

impl ConnecteurServer {
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
    
    pub fn get_catalog(&mut self) -> anyhow::Result<Vec<CatalogRow>> {
        let rows = self.client.query(GET_CATALOG, &[])?;
        let mut out = Vec::with_capacity(rows.len());
        for r in rows {
            out.push(CatalogRow {
                id: r.try_get(0)?,
                ean: r.try_get(1)?,
                family_id: r.try_get(2)?,
                height: r.try_get(3)?,
            });
        }
        Ok(out)
    }
}

//--------------------------------------------------------------------------------------------------
// End of file
//--------------------------------------------------------------------------------------------------