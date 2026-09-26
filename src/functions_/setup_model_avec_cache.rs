
use ort::session::Session;
use ort::ep::cuda::CUDA;
use ort::ep::tensorrt::TensorRT;

pub fn setup_model_avec_cache(
        // signature simplifiée pour l'exemple ; dans yolo_26.rs gardez
        // `&mut self` comme actuellement.
        model_path: &str,
        device: &str,
    ) -> anyhow::Result<ort::session::Session> {

        let mut builder = Session::builder()
            .map_err(|e| anyhow::anyhow!("Erreur création du builder : {}", e))?;

        if device.eq_ignore_ascii_case("cuda") {
            // Chemin persistant tant que le node de calcul reste up (fenêtre des
            // idle_time_before_scale_down=120s d'Azure ML). Lisez-le depuis une
            // variable d'environnement pour pouvoir pointer facilement vers un
            // chemin monté sur datastore si le disque local ne survit pas entre
            // deux jobs (à valider empiriquement, cf. message précédent).
            let engine_cache_dir = std::env::var("TRT_ENGINE_CACHE_DIR")
                .unwrap_or_else(|_| "/tmp/trt_engine_cache".to_string());
            std::fs::create_dir_all(&engine_cache_dir).ok();

            let cuda_provider = CUDA::default().build();

            let trt_provider = TensorRT::default()
                .with_engine_cache(true)
                .with_engine_cache_path(&engine_cache_dir)
                .build();

            builder = builder
                .with_execution_providers([trt_provider, cuda_provider])
                .map_err(|e| anyhow::anyhow!("Erreur configuration des execution providers : {}", e))?;
        }

        builder
            .commit_from_file(model_path)
            .map_err(|e| anyhow::anyhow!("Erreur chargement modèle '{}': {}", model_path, e))
    }