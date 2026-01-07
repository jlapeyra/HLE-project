#!/bin/bash
# Setup script for HLE-project Python environment

set -e

echo "=========================================="
echo "HLE-project Environment Setup"
echo "=========================================="

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo ""
echo "1. Creating virtual environment at: $VENV_DIR"
python3 -m venv "$VENV_DIR"

echo ""
echo "2. Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo ""
echo "3. Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

echo ""
echo "4. Installing dependencies from requirements.txt..."
pip install -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "=========================================="
echo "✓ Environment setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment in the future, run:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
echo ""
