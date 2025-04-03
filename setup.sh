#!/bin/bash

set -e  # Exit on any error

# === Utility Functions ===
show_help() {
    echo ""
    echo "Usage: bash scripts/setup.sh [command]"
    echo ""
    echo "Commands:"
    echo "  init         Initialize git submodules"
    echo "  update       Pull latest changes for submodules"
    echo "  status       Show submodule status"
    echo "  install      Install submodules as editable packages"
    echo "  venv         Create Python virtual environment (.venv)"
    echo "  clean        Remove Python build/cache files"
    echo ""
}

# === Commands ===

init_submodules() {
    echo "Initializing submodules..."
    git submodule update --init --recursive
}

update_submodules() {
    echo "Updating submodules..."
    git submodule foreach 'echo "Updating $name..." && git pull origin main'
}

status_submodules() {
    echo "Submodule status:"
    git submodule status
}

install_editable() {
    echo "Installing submodules as editable packages..."
    pip install -e ./causal_models
    pip install -e ./mab
}

create_venv() {
    echo "Creating virtual environment (.venv)..."
    python -m venv .venv
    echo "Virtual environment created. To activate:"
    echo "source .venv/bin/activate  # (Linux/Mac)"
    echo ".venv\\Scripts\\activate     # (Windows)"
}

clean_pycache() {
    echo "Cleaning __pycache__ and .pyc files..."
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete
    echo "Clean complete."
}

# === Entry Point ===

case "$1" in
    init) init_submodules ;;
    update) update_submodules ;;
    status) status_submodules ;;
    install) install_editable ;;
    venv) create_venv ;;
    clean) clean_pycache ;;
    help | *) show_help ;;
esac
