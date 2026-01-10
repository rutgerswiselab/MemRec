#!/bin/bash
# MemRec Environment Setup Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "  MemRec Environment Setup"
echo "=========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo ""
    echo "Please install Miniconda or Anaconda first:"
    echo ""
    echo "Option 1: Install Miniconda (recommended, lightweight)"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    echo ""
    echo "Option 2: Install Anaconda (full distribution)"
    echo "  wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh"
    echo "  bash Anaconda3-2024.02-1-Linux-x86_64.sh"
    echo ""
    echo "After installation, restart your shell and run this script again."
    echo ""
    exit 1
fi

echo "✓ Conda found: $(conda --version)"
echo ""

# Create conda environment
echo "Creating conda environment 'memrec'..."
cd "$PROJECT_ROOT"
conda env create -f env/environment.yml -y || {
    echo "Environment already exists. Updating..."
    conda env update -f env/environment.yml --prune
}

echo ""
echo "Environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate memrec"
echo ""
echo "Verifying PyTorch installation..."
conda run -n memrec python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'Device count: {torch.cuda.device_count()}')"
echo ""
echo "=========================================="
