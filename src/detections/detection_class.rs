use pyo3::prelude::*;

#[pyclass(eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DetectionClass {
    Etiquette,
    OtherEtiquette,
    Produit,
    NoProduct,
    Publicity,
    Reference
}

impl std::fmt::Display for DetectionClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DetectionClass::Etiquette => write!(f, "etiquette"),
            DetectionClass::OtherEtiquette => write!(f, "other_etiquette"),
            DetectionClass::Produit => write!(f, "product"),
            DetectionClass::NoProduct => write!(f, "no_product"),
            DetectionClass::Publicity => write!(f, "publicity"),
            DetectionClass::Reference => write!(f, "reference"),
        }
    }
}

impl std::str::FromStr for DetectionClass {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "etiquette" => Ok(DetectionClass::Etiquette),
            "other_etiquette" => Ok(DetectionClass::OtherEtiquette),
            "product" => Ok(DetectionClass::Produit),
            "noproduct" => Ok(DetectionClass::NoProduct),
            "publicity" => Ok(DetectionClass::Publicity),
            "reference" => Ok(DetectionClass::Reference),
            other => Err(format!("Catégorie inconnue: {other}")),
        }
    }
}