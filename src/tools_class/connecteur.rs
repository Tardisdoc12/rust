//--------------------------------------------------------------------------------------------------
// Filename: connecteur.rs
// Author: Jean Anquetil
// Date: 2026-10-03
//--------------------------------------------------------------------------------------------------

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

impl ConnecteurServer {
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