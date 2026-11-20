use ort::session::Session;
use ort::ep::ArenaExtendStrategy;
use ort::ep::cuda::{CUDA, ConvAlgorithmSearch};

pub fn setup_model_avec_cache(
    model_path: &str,
    device: &str,
) -> anyhow::Result<ort::session::Session> {
    let mut builder = Session::builder()
        .map_err(|e| anyhow::anyhow!("Erreur création du builder : {}", e))?;

    if device.eq_ignore_ascii_case("cuda") {
        let cuda_provider = CUDA::default()
            .with_arena_extend_strategy(ArenaExtendStrategy::SameAsRequested) // pas de réserve par puissances de 2
            .with_conv_algorithm_search(ConvAlgorithmSearch::Heuristic)       // nom corrigé pour rc.13
            .with_conv_max_workspace(false)                                    // espace temporaire cuDNN limité
            .build();

        builder = builder
            .with_execution_providers([cuda_provider])
            .map_err(|e| anyhow::anyhow!("Erreur configuration des execution providers : {}", e))?;
    }

    builder
        .commit_from_file(model_path)
        .map_err(|e| anyhow::anyhow!("Erreur chargement modèle '{}': {}", model_path, e))
}
