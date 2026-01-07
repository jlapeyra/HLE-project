# Environment Setup Guide

This guide explains how to set up and manage the Python virtual environment for the HLE-project.

## Quick Start (Recommended)

### Using the automated setup script:

```bash
cd /mnt/c/Users/chatbot/Desktop/hle/HLE-project
bash setup_env.sh
```

This script will:
1. Create a Python virtual environment
2. Activate it
3. Upgrade pip, setuptools, and wheel
4. Install all dependencies from `requirements.txt`

## Manual Setup

If you prefer to set up manually:

### 1. Create the virtual environment:
```bash
cd /mnt/c/Users/chatbot/Desktop/hle/HLE-project
python3 -m venv .venv
```

### 2. Activate the virtual environment:
```bash
source .venv/bin/activate
```

### 3. Upgrade pip and build tools:
```bash
pip install --upgrade pip setuptools wheel
```

### 4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

The project requires:
- **torch** - Deep learning framework
- **torch-geometric** - Graph neural network library
- **networkx** - Graph analysis library
- **numpy** - Numerical computing
- **tqdm** - Progress bars

## Using the Environment

### Activate the environment:
```bash
source .venv/bin/activate
```

### Run Python scripts:
```bash
python src/g2g_nn.py
python src/nn_train.py
python src/nn_test.py
```

### Deactivate the environment:
```bash
deactivate
```

## Checking Installation

To verify the environment is set up correctly:

```bash
source .venv/bin/activate
python -c "import torch; import torch_geometric; import networkx; print('✓ All dependencies installed')"
```

## Installing Additional Packages

If you need to install more packages:

```bash
source .venv/bin/activate
pip install <package_name>
```

To update `requirements.txt` with new packages:

```bash
pip freeze > requirements.txt
```

## Troubleshooting

### Virtual environment not activating?
Make sure you're in the correct directory and the `.venv` folder exists.

### Permission denied on setup_env.sh?
Run:
```bash
chmod +x setup_env.sh
bash setup_env.sh
```

### Dependency installation fails?
Update pip first:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
