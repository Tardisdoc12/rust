#!/usr/bin/env bash
# Construit la wheel Python de la lib Rust, avec les modules ONNX Runtime (CUDA) embarqués.
# Usage : ./build_wheel.sh [chemin/vers/python]   (par défaut : /home/jean/venv/bin/python)
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${1:-/home/jean/venv/bin/python}"
PKG_DIR="python/rust"

[ -f "$PKG_DIR/__init__.py" ] || { echo "manquant : $PKG_DIR/__init__.py" >&2; exit 1; }

# Contournement du bug ort-sys (nouvelle organisation de target/ dans cargo récent)
mkdir -p target/release/build/deps target/release/build/examples

# 1) Compile : ort-sys télécharge ONNX Runtime et dépose les modules dans target/
#    PYO3_BUILD_EXTENSION_MODULE : compile comme une extension (pas de lien à libpython)
PYO3_BUILD_EXTENSION_MODULE=1 PYO3_PYTHON="$PYTHON" cargo build --release --lib

# 2) Copie les modules dans le paquet Python
for lib in libonnxruntime_providers_shared.so libonnxruntime_providers_cuda.so; do
    src=$(find target/release -maxdepth 2 -name "$lib" -print -quit)
    [ -n "$src" ] || { echo "introuvable : $lib" >&2; exit 1; }
    cp "$src" "$PKG_DIR/"
done

# 3) Construit la wheel
rm -f target/wheels/rust-*.whl
maturin build --release --interpreter "$PYTHON"

# 4) Vérifie que la wheel contient bien tout
WHEEL=$(ls target/wheels/rust-*.whl)
for f in rust/__init__.py rust/libonnxruntime_providers_shared.so rust/libonnxruntime_providers_cuda.so; do
    unzip -l "$WHEEL" | grep -q " $f$" || { echo "absent de la wheel : $f (vérifie .gitignore)" >&2; exit 1; }
done
echo "OK : $WHEEL"
