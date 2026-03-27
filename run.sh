#!/usr/bin/env bash
set -e

if ! command -v python3 &>/dev/null; then
    echo "Python 3 not found. Please install it first."
    exit 1
fi

python3 -c "from PIL import Image" 2>/dev/null || pip3 install Pillow -q
python3 -c "import numpy"          2>/dev/null || pip3 install numpy  -q
python3 -c "import numba"          2>/dev/null || { echo "Installing numba (one-time, ~1 min)..."; pip3 install numba -q; }

python3 substrate_gui.py
