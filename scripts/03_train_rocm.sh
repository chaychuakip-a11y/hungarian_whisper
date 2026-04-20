#!/bin/bash
# Train LoRA Whisper model on Hungarian speech data with ROCm (AMD GPU)
# Note: Requires PyTorch built with ROCm support

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "========================================"
echo "Training Hungarian Whisper Model (ROCm)"
echo "========================================"

# Check ROCm
echo "Checking ROCm environment..."
if ! command -v rocm-smi &> /dev/null; then
    echo "ERROR: rocm-smi not found. ROCm not installed."
    exit 1
fi

rocm-smi --showproductname
echo ""

# Check PyTorch ROCm
python3 << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"ROCm available: {hasattr(torch.version, 'hip') and torch.version.hip is not None}")
if hasattr(torch.version, 'hip'):
    print(f"ROCm version: {torch.version.hip}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
EOF

echo ""

python3 << 'EOF'
import sys
import logging
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
import torch
import json

from model.lora_whisper import create_lora_whisper
from data.dataset_loader import HungarianDatasetLoader
from data.collator import DataCollatorSpeechSeq2SeqWithPadding
from training.trainer import train_whisper
from training.evaluation import create_compute_metrics_func
from utils.memory_monitor import print_memory_summary, MemoryMonitor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check ROCm device
if not torch.cuda.is_available():
    logger.warning("ROCm/CUDA not available!")
    # Try to use AMD GPU via HIP
    try:
        import torch.cuda as cuda
        logger.info("Attempting HIP backend...")
    except:
        pass

# Load ROCm config
config_path = "config/config_rocm.yaml"
if not Path(config_path).exists():
    config_path = "config/config.yaml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

logger.info("Loading datasets...")
loader = HungarianDatasetLoader(
    cache_dir="./data",
    min_duration=config["data"]["filters"]["min_duration"],
    max_duration=config["data"]["filters"]["max_duration"]
)

train_dataset, eval_dataset = loader.load_all_datasets(
    train_ratio=config["data"].get("train_split", 0.9)
)

# Use subset for faster iteration
MAX_TRAIN_SAMPLES = 2000
MAX_EVAL_SAMPLES = 200
if len(train_dataset) > MAX_TRAIN_SAMPLES:
    train_dataset = train_dataset.select(range(MAX_TRAIN_SAMPLES))
if len(eval_dataset) > MAX_EVAL_SAMPLES:
    eval_dataset = eval_dataset.select(range(MAX_EVAL_SAMPLES))

logger.info(f"Using subset - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

# Load model with LoRA
logger.info("Loading Whisper model with LoRA...")
model_name = config["model"]["name"]
lora_config = config["model"]["lora"]
int8 = config["model"].get("int8", False)  # Disable INT8 for ROCm

logger.info(f"Model: {model_name}")
logger.info(f"LoRA config: {lora_config}")
logger.info(f"INT8: {int8} (disabled for ROCm compatibility)")

try:
    model, feature_extractor, tokenizer = create_lora_whisper(
        model_name=model_name,
        lora_r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        target_modules=lora_config["target_modules"],
        int8=int8
    )
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    logger.error("ROCm may not be supported by this PyTorch build.")
    logger.info("To enable ROCm support:")
    logger.info("  pip install torch --index-url https://download.pytorch.org/whl/rocm5.7")
    sys.exit(1)

# Create data collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    feature_extractor=feature_extractor,
    tokenizer=tokenizer,
    padding=True
)

# Compute metrics function
compute_metrics = create_compute_metrics_func(tokenizer)

# Create output directory
output_dir = config["training"]["output_dir"]
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Train
logger.info("Starting training...")
trainer, train_result = train_whisper(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    feature_extractor=feature_extractor,
    tokenizer=tokenizer,
    data_collator=data_collator,
    output_dir=output_dir,
    per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
    gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
    learning_rate=config["training"]["learning_rate"],
    num_train_epochs=1,  # Just 1 epoch for testing
    warmup_steps=config["training"]["warmup_steps"],
    logging_steps=config["training"]["logging_steps"],
    save_steps=config["training"]["save_steps"],
    eval_steps=config["training"]["eval_steps"],
    bf16=config["training"].get("bf16", True),
    fp16=config["training"].get("fp16", False),
    compute_metrics=compute_metrics,
    max_vram_gb=config["training"].get("max_vram_gb", 24.0)
)

logger.info("Training complete!")
metrics = trainer.evaluate()
logger.info(f"Final evaluation metrics: {metrics}")

print_memory_summary()

EOF

echo "========================================"
echo "ROCm training complete"
echo "========================================"
