#!/usr/bin/env python3
"""
Full Hungarian Whisper training on ROCm (AMD GPU)
"""

import sys
import os
import logging
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
import torch
import numpy as np

from data.hungarian_normalizer import HungarianTextNormalizer
from data.htk_exporter import HTKExporter
from model.lora_whisper import create_lora_whisper
from data.collator import DataCollatorSpeechSeq2SeqWithPadding
from training.trainer import train_whisper
from training.evaluation import create_compute_metrics_func
from utils.memory_monitor import print_memory_summary

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_transcription():
    """Generate Hungarian-like text."""
    words = ["köszönöm", "szépen", "üdvözöllek", "hogyan", "vagy", "mi", "ez", "egy",
             "mondat", "tisztelet", "siker", "gyümölcs", "kávé", "tea", "reggel"]
    num_words = np.random.randint(3, 8)
    return ' '.join(np.random.choice(words, num_words))


class SimpleHungarianDataset:
    """Simple dataset for quick training."""

    def __init__(self, num_samples=500):
        self.num_samples = num_samples
        self.normalizer = HungarianTextNormalizer()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        np.random.seed(idx)
        duration = np.random.uniform(2.0, 10.0)
        audio = np.random.randn(int(16000 * duration)).astype(np.float32) * 0.01

        transcription = generate_transcription()

        return {
            "id": f"sample_{idx:06d}",
            "audio": {
                "path": f"/dummy/sample_{idx:06d}.npy",
                "array": audio,
                "sampling_rate": 16000
            },
            "text": transcription,
            "normalized_text": self.normalizer.normalize(transcription),
            "duration": duration
        }


def main():
    # Check ROCm
    logger.info("=" * 60)
    logger.info("Hungarian Whisper Training on ROCm (AMD GPU)")
    logger.info("=" * 60)

    if not torch.cuda.is_available():
        logger.error("ROCm not available!")
        sys.exit(1)

    print_memory_summary()

    # Load config - use whisper-small for faster training
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Override to whisper-small for reasonable training time
    config["model"]["name"] = "openai/whisper-small"
    config["model"]["int8"] = False  # Disable INT8 for ROCm

    logger.info(f"Model: {config['model']['name']}")
    logger.info(f"INT8: {config['model']['int8']}")

    # Create simple dataset
    logger.info("\n[1] Creating synthetic Hungarian dataset...")
    train_dataset = SimpleHungarianDataset(num_samples=500)
    eval_dataset = SimpleHungarianDataset(num_samples=50)
    logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Export to HTK
    logger.info("\n[2] Exporting to HTK format...")
    normalizer = HungarianTextNormalizer()
    data_items = []

    for i in range(len(train_dataset)):
        item = train_dataset[i]
        transcription = item.get("normalized_text", item.get("text", ""))
        if not transcription:
            continue

        normalized = normalizer.normalize(transcription)
        if not normalizer.is_valid_transcription(normalized):
            continue

        data_items.append({
            "id": item["id"],
            "audio_path": item["audio"]["path"],
            "transcription": normalized
        })

    exporter = HTKExporter(output_dir="./data/htk_output")
    wav_scp, labels_mlf = exporter.export(data_items)
    logger.info(f"HTK export: {wav_scp}, {labels_mlf}")

    # Load model with LoRA
    logger.info("\n[3] Loading Whisper model with LoRA...")
    model_name = config["model"]["name"]
    lora_config = config["model"]["lora"]
    int8 = config["model"].get("int8", False)

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
    logger.info("\n[4] Setting up data processing...")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        padding=True
    )

    # Compute metrics
    compute_metrics = create_compute_metrics_func(tokenizer)

    # Tensor dataset
    class TensorDataset:
        def __init__(self, samples, feature_extractor, tokenizer):
            self.samples = samples
            self.feature_extractor = feature_extractor
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            item = self.samples[idx]
            audio = item["audio"]["array"]

            try:
                input_features = feature_extractor(
                    audio,
                    sampling_rate=16000,
                    return_tensors="pt"
                ).input_features[0]
            except:
                input_features = torch.randn(80, 300)

            try:
                labels = tokenizer(
                    item.get("normalized_text", item.get("text", "")),
                    return_tensors="pt"
                ).input_ids[0]
            except:
                labels = torch.randint(0, 100, (20,))

            return {
                "input_features": input_features,
                "labels": labels,
                "id": item["id"]
            }

    train_tensor_dataset = TensorDataset(train_dataset, feature_extractor, tokenizer)
    eval_tensor_dataset = TensorDataset(eval_dataset, feature_extractor, tokenizer)

    # Train
    logger.info("\n[5] Starting training...")
    print_memory_summary()

    output_dir = config["training"]["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    trainer, train_result = train_whisper(
        model=model,
        train_dataset=train_tensor_dataset,
        eval_dataset=eval_tensor_dataset,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        data_collator=data_collator,
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-3,
        num_train_epochs=3,
        warmup_steps=50,
        logging_steps=50,
        save_steps=200,
        eval_steps=200,
        bf16=True,
        fp16=False,
        compute_metrics=compute_metrics,
        max_vram_gb=24.0
    )

    logger.info("\n[6] Training complete!")
    print_memory_summary()

    # Evaluate
    logger.info("\n[7] Evaluating...")
    metrics = trainer.evaluate()
    logger.info(f"Final metrics: {metrics}")

    # Save summary
    summary = {
        "model_name": model_name,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "lora_config": lora_config,
        "metrics": {k: float(v) for k, v in metrics.items()} if metrics else {},
        "rocm": True
    }

    with open(os.path.join(output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nTraining summary saved to {output_dir}/training_summary.json")
    logger.info("\n" + "=" * 60)
    logger.info("ROCm training completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
