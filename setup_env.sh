#!/bin/bash
set -e

if nvidia-smi &>/dev/null; then
    CUDA_VER=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | tr -d .)
    echo "CUDA ${CUDA_VER} detected — installing GPU torch"
    uv sync
    uv pip install torch torch-geometric \
        --index-url "https://download.pytorch.org/whl/cu${CUDA_VER}"
elif [[ "$(uname)" == "Darwin" ]]; then
    echo "macOS detected — installing CPU/MPS torch"
    uv sync
else
    echo "No GPU detected — installing CPU torch"
    uv sync
    uv pip install torch torch-geometric \
        --index-url "https://download.pytorch.org/whl/cpu"
fi

echo ""
echo "Done. Verifying torch:"
uv run python -c "import torch; print(f'torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, MPS available: {torch.backends.mps.is_available()}')"
