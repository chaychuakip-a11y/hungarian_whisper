#!/bin/bash
# Train LoRA Whisper model on Hungarian speech data
# Adapted from company's Hulk framework training approach

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "========================================"
echo "Training Hungarian Whisper Model"
echo "========================================"

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

# Check CUDA
if not torch.cuda.is_available():
    logger.error("CUDA not available! Training requires GPU.")
    sys.exit(1)

print_memory_summary()

# Load config
with open("config/config.yaml", "r") as f:
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

logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

# Use a smaller subset for initial training run
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
int8 = config["model"].get("int8", True)

model, feature_extractor, tokenizer = create_lora_whisper(
    model_name=model_name,
    lora_r=lora_config["r"],
    lora_alpha=lora_config["lora_alpha"],
    lora_dropout=lora_config["lora_dropout"],
    target_modules=lora_config["target_modules"],
    int8=int8
)

logger.info("Model loaded successfully")

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
    num_train_epochs=config["training"]["num_train_epochs"],
    warmup_steps=config["training"]["warmup_steps"],
    logging_steps=config["training"]["logging_steps"],
    save_steps=config["training"]["save_steps"],
    eval_steps=config["training"]["eval_steps"],
    bf16=config["training"].get("bf16", True),
    fp16=config["training"].get("fp16", False),
    compute_metrics=compute_metrics,
    max_vram_gb=24.0
)

logger.info("Training complete!")

# Save final metrics
metrics = trainer.evaluate()
logger.info(f"Final evaluation metrics: {metrics}")

# Save training summary
summary = {
    "model_name": model_name,
    "train_samples": MAX_TRAIN_SAMPLES,
    "eval_samples": MAX_EVAL_SAMPLES,
    "lora_config": lora_config,
    "training_config": {
        "per_device_train_batch_size": config["training"]["per_device_train_batch_size"],
        "gradient_accumulation_steps": config["training"]["gradient_accumulation_steps"],
        "learning_rate": config["training"]["learning_rate"],
        "num_train_epochs": config["training"]["num_train_epochs"]
    },
    "metrics": {k: float(v) for k, v in metrics.items()} if metrics else {}
}

with open(os.path.join(output_dir, "training_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print_memory_summary()

logger.info("Training pipeline completed!")
logger.info(f"Checkpoints saved to: {output_dir}")

EOF

echo "========================================"
echo "Training complete"
echo "========================================"
