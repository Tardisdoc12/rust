#!/usr/bin/env bash
# Installe les bibliothèques CUDA 13.0 + cuDNN 9 nécessaires à ONNX Runtime (CUDAExecutionProvider).
# - N'installe PAS le pilote NVIDIA (il doit déjà être présent : nvidia-smi doit fonctionner).
# - N'installe que les bibliothèques utiles (pas tout le toolkit) pour économiser le disque.
# - Peut être relancé sans risque.
set -euo pipefail

CUDA_MAJOR_MINOR="13-0"            # paquets *-13-0
CUDNN_VERSION="9.27.0.42-1"        # même version que dia-dev-ia ; vide = dernière disponible
MIN_FREE_GB=3

log() { echo -e "\033[1;34m[cuda]\033[0m $*"; }
die() { echo -e "\033[1;31m[cuda] ERREUR :\033[0m $*" >&2; exit 1; }

# --- 1. Prérequis ----------------------------------------------------------
command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1 \
    || die "pilote NVIDIA absent ou GPU non détecté (nvidia-smi échoue)"

. /etc/os-release
[ "$ID" = "ubuntu" ] || die "script prévu pour Ubuntu (détecté : $ID)"
REPO="ubuntu${VERSION_ID//./}"     # 24.04 -> ubuntu2404

free_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
[ "$free_gb" -ge "$MIN_FREE_GB" ] || die "seulement ${free_gb} Go libres sur /, il en faut au moins ${MIN_FREE_GB}"

# --- 2. Dépôt CUDA de NVIDIA -----------------------------------------------
if ! apt-cache policy "cuda-cudart-${CUDA_MAJOR_MINOR}" | grep -q 'Candidate: [0-9]'; then
    log "ajout du dépôt CUDA NVIDIA ($REPO)"
    tmp=$(mktemp -d)
    wget -q -O "$tmp/cuda-keyring.deb" \
        "https://developer.download.nvidia.com/compute/cuda/repos/${REPO}/x86_64/cuda-keyring_1.1-1_all.deb"
    sudo dpkg -i "$tmp/cuda-keyring.deb"
    rm -rf "$tmp"
    sudo apt-get update
else
    log "dépôt CUDA déjà configuré"
fi

# --- 3. Bibliothèques ------------------------------------------------------
PKGS=(
    "cuda-cudart-${CUDA_MAJOR_MINOR}"
    "libcublas-${CUDA_MAJOR_MINOR}"
    "libcufft-${CUDA_MAJOR_MINOR}"
    "libcurand-${CUDA_MAJOR_MINOR}"
    "libnvjitlink-${CUDA_MAJOR_MINOR}"
    "cuda-nvrtc-${CUDA_MAJOR_MINOR}"
    "libcudnn9-cuda-13${CUDNN_VERSION:+=$CUDNN_VERSION}"
)
log "installation : ${PKGS[*]}"
sudo apt-get install -y --no-install-recommends "${PKGS[@]}"

# Marque-les comme installés volontairement : 'apt autoremove' ne les supprimera pas
sudo apt-mark manual "${PKGS[@]%%=*}" >/dev/null
# Empêche une mise à jour automatique de cuDNN vers une version différente
[ -n "$CUDNN_VERSION" ] && sudo apt-mark hold libcudnn9-cuda-13 >/dev/null

sudo apt-get clean
sudo ldconfig

# --- 4. Vérification -------------------------------------------------------
missing=0
for lib in libcudart.so libcublas.so libcublasLt.so libcufft.so libcurand.so libcudnn.so; do
    if ldconfig -p | grep -q "$lib"; then
        log "OK      $lib"
    else
        echo "        ABSENT  $lib"; missing=1
    fi
done
[ "$missing" -eq 0 ] || die "certaines bibliothèques sont introuvables"

PROVIDER=$(python3 - 2>/dev/null <<'EOF' || true
import glob; print(next(iter(glob.glob("/home/*/venv/lib/python3*/site-packages/rust/libonnxruntime_providers_cuda.so")), ""))
EOF
)
if [ -n "$PROVIDER" ]; then
    if ldd "$PROVIDER" | grep -q "not found"; then
        ldd "$PROVIDER" | grep "not found"
        die "le module CUDA d'ONNX Runtime a des dépendances manquantes"
    fi
    log "module ONNX Runtime CUDA : toutes les dépendances sont trouvées"
fi

log "terminé — $(df -h --output=avail / | tail -1 | tr -d ' ') libres sur /"
