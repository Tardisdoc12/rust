
use ort::session::Session;
use ort::ep::cuda::CUDA;

pub fn setup_model_avec_cache(
        // signature simplifiée pour l'exemple ; dans yolo_26.rs gardez
        // `&mut self` comme actuellement.
        model_path: &str,
        device: &str,
    ) -> anyhow::Result<ort::session::Session> {

        let mut builder = Session::builder()
            .map_err(|e| anyhow::anyhow!("Erreur création du builder : {}", e))?;

        if device.eq_ignore_ascii_case("cuda") {
            let cuda_provider = CUDA::default().build();

            builder = builder
                .with_execution_providers([cuda_provider])
                .map_err(|e| anyhow::anyhow!("Erreur configuration des execution providers : {}", e))?;
        }

        builder
            .commit_from_file(model_path)
            .map_err(|e| anyhow::anyhow!("Erreur chargement modèle '{}': {}", model_path, e))
    }