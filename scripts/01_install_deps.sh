#!/bin/bash
# Install dependencies for Hungarian Whisper fine-tuning pipeline

set -e

echo "========================================"
echo "Installing Python dependencies"
echo "========================================"

pip install --upgrade pip

pip install -r requirements.txt

echo "========================================"
echo "Installing HTK toolkit (if available)"
echo "========================================"
# HTK is optional - the pipeline can work without it for non-HTK workflows
if command -v apt-get &> /dev/null; then
    echo "Note: HTK will be used if available for format conversion"
fi

echo "========================================"
echo "Dependency installation complete"
echo "========================================"
