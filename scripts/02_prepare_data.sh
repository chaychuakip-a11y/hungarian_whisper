#!/bin/bash
# Prepare Hungarian datasets for Whisper training
# Loads from HuggingFace, filters, normalizes, and exports to HTK/LMDB format

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "========================================"
echo "Preparing Hungarian Speech Datasets"
echo "========================================"

python3 << 'EOF'
import sys
import logging
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.dataset_loader import HungarianDatasetLoader
from data.hungarian_normalizer import HungarianTextNormalizer
from data.htk_exporter import HTKExporter
from data.lmdb_preparator import prepare_hungarian_data
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Load config
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

print("Loading Hungarian datasets...")
loader = HungarianDatasetLoader(
    cache_dir="./data",
    min_duration=config["data"]["filters"]["min_duration"],
    max_duration=config["data"]["filters"]["max_duration"]
)

train_dataset, eval_dataset = loader.load_all_datasets(
    train_ratio=config["data"].get("train_split", 0.9)
)

print(f"Train samples: {len(train_dataset)}")
print(f"Eval samples: {len(eval_dataset)}")

# Use a subset for faster processing (for initial run)
MAX_SAMPLES = 10000
if len(train_dataset) > MAX_SAMPLES:
    train_dataset = train_dataset.select(range(MAX_SAMPLES))
    print(f"Using subset: {MAX_SAMPLES} train samples")

# Step 1: Export to HTK format
print("\n[Step 1] Converting to HTK format...")

normalizer = HungarianTextNormalizer()
data_items = []

for i, item in enumerate(train_dataset):
    audio_path = item["audio"]["path"]
    transcription = item.get("normalized_text", item.get("text", ""))

    if not transcription:
        continue

    # Validate normalization
    normalized = normalizer.normalize(transcription)
    if not normalizer.is_valid_transcription(normalized):
        continue

    data_items.append({
        "id": f"sample_{i:06d}",
        "audio_path": audio_path,
        "transcription": normalized
    })

print(f"Exporting {len(data_items)} samples to HTK format...")

exporter = HTKExporter(output_dir="./data/htk_output")
wav_scp, labels_mlf = exporter.export(data_items)

print(f"HTK export complete!")
print(f"  wav.scp: {wav_scp}")
print(f"  labels.mlf: {labels_mlf}")

# Step 2: Prepare LMDB format for Hulk training
print("\n[Step 2] Preparing LMDB format...")

# Hungarian phone dictionary (simplified - for ED mode we use text)
phone2idx = None  # Use text labels for ED mode

lmdb_output_dir = "./data/lmdb_output"
Path(lmdb_output_dir).mkdir(parents=True, exist_ok=True)

# Prepare data using the HF datasets directly
lmdb_paths = prepare_hungarian_data(
    output_dir=lmdb_output_dir,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    phone2idx=phone2idx,
    chunk_size=config["training"].get("chunk_size", 10000)
)

print(f"LMDB preparation complete!")
for key, path in lmdb_paths.items():
    print(f"  {key}: {path}")

# Save dataset info
dataset_info = {
    "train_samples": len(train_dataset),
    "eval_samples": len(eval_dataset),
    "exported_samples": len(data_items),
    "htk_export": {
        "wav_scp": wav_scp,
        "labels_mlf": labels_mlf
    },
    "lmdb_paths": lmdb_paths,
    "config": {
        "min_duration": config["data"]["filters"]["min_duration"],
        "max_duration": config["data"]["filters"]["max_duration"],
        "train_split": config["data"].get("train_split", 0.9)
    }
}

with open("./data/dataset_info.json", "w") as f:
    json.dump(dataset_info, f, indent=2)

print("\n========================================")
print("Dataset preparation complete!")
print("========================================")
print(f"Summary:")
print(f"  - Train samples: {len(train_dataset)}")
print(f"  - Eval samples: {len(eval_dataset)}")
print(f"  - HTK files: {wav_scp}, {labels_mlf}")
print(f"  - LMDB files: {lmdb_output_dir}")

EOF
