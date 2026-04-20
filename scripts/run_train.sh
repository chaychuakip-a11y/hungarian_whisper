#!/bin/bash
# Run Hulk-style training without proxy issues

# Clear all proxy settings
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
unset all_proxy ALL_PROXY socks_proxy SOCKS_PROXY

# Activate rocm_env environment (ROCm 6.1 with PyTorch 2.6)
source /home/lty/anaconda3/bin/activate rocm_env

# Go to project directory
cd /home/lty/hungarian_whisper

# Run training
python scripts/train_rocm_hulk_style.py