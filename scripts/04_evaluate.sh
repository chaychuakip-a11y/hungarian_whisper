#!/bin/bash
# Evaluate trained Hungarian Whisper model

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "========================================"
echo "Evaluating Hungarian Whisper Model"
echo "========================================"

python3 << 'EOF'
import sys
import logging
from pathlib import Path
import yaml

import torch
from transformers import WhisperForConditionalGeneration

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.dataset_loader import HungarianDatasetLoader
from training.evaluation import EvaluationRunner
from utils.memory_monitor import print_memory_summary

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check CUDA
if not torch.cuda.is_available():
    logger.error("CUDA not available!")
    sys.exit(1)

# Load config
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load datasets
logger.info("Loading datasets...")
loader = HungarianDatasetLoader(
    cache_dir="./data",
    min_duration=config["data"]["filters"]["min_duration"],
    max_duration=config["data"]["filters"]["max_duration"]
)

train_dataset, eval_dataset = loader.load_all_datasets(train_ratio=0.9)

# Limit samples for evaluation
MAX_EVAL_SAMPLES = 1000
if len(eval_dataset) > MAX_EVAL_SAMPLES:
    eval_dataset = eval_dataset.select(range(MAX_EVAL_SAMPLES))

logger.info(f"Evaluating on {len(eval_dataset)} samples")

# Load model
checkpoint_path = Path(config["training"]["output_dir"])
best_checkpoint = checkpoint_path / "best_model"

if best_checkpoint.exists():
    logger.info(f"Loading best checkpoint from {best_checkpoint}")
    model = WhisperForConditionalGeneration.from_pretrained(str(best_checkpoint))
else:
    logger.warning("No best checkpoint found, using final model")
    model = WhisperForConditionalGeneration.from_pretrained(
        config["model"]["name"]
    )

model = model.to("cuda")
model.eval()

# Load processor
from transformers import WhisperProcessor
processor = WhisperProcessor.from_pretrained(config["model"]["name"])
tokenizer = processor.tokenizer

# Run evaluation
runner = EvaluationRunner(tokenizer=tokenizer)

# Prepare evaluation samples
sample_texts = []
ref_texts = []

for item in eval_dataset:
    sample_texts.append({
        "input_features": item["input_features"],
        "labels": item["labels"]
    })
    ref_texts.append(item.get("normalized_text", item.get("text", "")))

# Simple evaluation using trainer
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./output/eval",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=448,
    fp16=True,
    report_to="none"
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    eval_dataset=sample_texts,
    tokenizer=tokenizer
)

logger.info("Running inference...")
predictions = trainer.predict(sample_texts, metric_key_prefix="eval")

# Decode predictions
decoded_preds = tokenizer.batch_decode(
    predictions.predictions,
    skip_special_tokens=True
)
decoded_preds = [p.lower().strip() for p in decoded_preds]

# Compute metrics
from evaluate import load
wer_metric = load("wer")
wer = wer_metric.compute(
    predictions=decoded_preds,
    references=ref_texts
)

logger.info(f"Word Error Rate: {wer:.4f}")
logger.info(f"Accuracy: {1 - wer:.4f}")

# Sample predictions
logger.info("\nSample predictions:")
for i in range(min(5, len(decoded_preds))):
    logger.info(f"Reference: {ref_texts[i][:100]}...")
    logger.info(f"Predicted: {decoded_preds[i][:100]}...")
    logger.info("---")

print_memory_summary()

EOF

echo "========================================"
echo "Evaluation complete"
echo "========================================"
