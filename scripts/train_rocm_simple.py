#!/usr/bin/env python3
"""
Simplified Hungarian Whisper training on ROCm (AMD GPU)
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
from dataclasses import dataclass

from data.hungarian_normalizer import HungarianTextNormalizer
from data.htk_exporter import HTKExporter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_transcription():
    """Generate Hungarian-like text."""
    words = ["köszönöm", "szépen", "üdvözöllek", "hogyan", "vagy", "mi", "ez", "egy",
             "mondat", "tisztelet", "siker", "gyümölcs", "kávé", "tea", "reggel"]
    num_words = np.random.randint(3, 8)
    return ' '.join(np.random.choice(words, num_words))


@dataclass
class SimpleSpeechSample:
    """Simple speech sample for training."""
    input_features: torch.Tensor
    labels: torch.Tensor


class SimpleDataset:
    """Simple dataset for quick training."""

    def __init__(self, num_samples=500, feature_extractor=None, tokenizer=None):
        self.num_samples = num_samples
        self.normalizer = HungarianTextNormalizer()
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

        # Pre-generate all samples
        self.samples = []
        for i in range(num_samples):
            np.random.seed(i)
            duration = np.random.uniform(2.0, 10.0)
            audio = np.random.randn(int(16000 * duration)).astype(np.float32) * 0.01
            transcription = generate_transcription()

            self.samples.append({
                "audio": audio,
                "transcription": transcription,
                "duration": duration
            })

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        item = self.samples[idx]
        audio = item["audio"]
        transcription = item["transcription"]

        # Extract features
        if self.feature_extractor:
            input_features = self.feature_extractor(
                audio,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features[0]
        else:
            input_features = torch.randn(80, 300)

        # Tokenize
        if self.tokenizer:
            labels = self.tokenizer(
                transcription,
                return_tensors="pt"
            ).input_ids[0]
        else:
            labels = torch.randint(0, 100, (20,))

        return SimpleSpeechSample(
            input_features=input_features,
            labels=labels
        )


def main():
    logger.info("=" * 60)
    logger.info("Hungarian Whisper Training on ROCm (AMD GPU)")
    logger.info("=" * 60)

    if not torch.cuda.is_available():
        logger.error("ROCm not available!")
        sys.exit(1)

    # Check memory
    allocated = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"VRAM: {allocated:.2f}GB / {total:.2f}GB")

    # Load model
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    logger.info("\n[1] Loading Whisper model...")
    model_name = "openai/whisper-small"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # Move to GPU
    device = torch.device("cuda")
    model = model.to(device)

    # Check memory after model load
    allocated = torch.cuda.memory_allocated() / 1e9
    logger.info(f"VRAM after model load: {allocated:.2f}GB")

    # Create dataset
    logger.info("\n[2] Creating dataset...")
    train_dataset = SimpleDataset(500, processor.feature_extractor, processor.tokenizer)
    eval_dataset = SimpleDataset(50, processor.feature_extractor, processor.tokenizer)

    # Export to HTK
    logger.info("\n[3] Exporting to HTK format...")
    data_items = []
    normalizer = HungarianTextNormalizer()
    for i, item in enumerate(train_dataset.samples[:100]):
        data_items.append({
            "id": f"sample_{i:06d}",
            "audio_path": f"/dummy/sample_{i:06d}.npy",
            "transcription": normalizer.normalize(item["transcription"])
        })

    exporter = HTKExporter(output_dir="./data/htk_output")
    wav_scp, labels_mlf = exporter.export(data_items)
    logger.info(f"HTK export: {wav_scp}, {labels_mlf}")

    # Training
    logger.info("\n[4] Starting training...")
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Simple training loop
    batch_size = 4
    num_epochs = 1
    max_steps = 50  # Limit steps for quick test

    step = 0
    for epoch in range(num_epochs):
        model.train()
        for i in range(0, len(train_dataset), batch_size):
            if step >= max_steps:
                break

            # Get batch
            batch_features = []
            batch_labels = []
            for j in range(i, min(i + batch_size, len(train_dataset))):
                sample = train_dataset[j]
                batch_features.append(sample.input_features)
                batch_labels.append(sample.labels)

            # Pad sequences
            batch_features = torch.nn.utils.rnn.pad_sequence(batch_features, batch_first=True)
            batch_labels = torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True)

            # Move to device
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            # Forward
            outputs = model(
                input_features=batch_features,
                labels=batch_labels
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % 10 == 0:
                allocated = torch.cuda.memory_allocated() / 1e9
                logger.info(f"Step {step}/{max_steps} | Loss: {loss.item():.4f} | VRAM: {allocated:.2f}GB")

            step += 1

        if step >= max_steps:
            break

    # Final memory
    allocated = torch.cuda.memory_allocated() / 1e9
    logger.info(f"\nFinal VRAM: {allocated:.2f}GB")

    # Save model
    logger.info("\n[5] Saving model...")
    output_dir = "./output/checkpoints"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    # Save summary
    summary = {
        "model_name": model_name,
        "train_samples": len(train_dataset),
        "max_steps": max_steps,
        "final_loss": float(loss.item()),
        "rocm": True
    }
    with open(os.path.join(output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nModel saved to {output_dir}")
    logger.info("\n" + "=" * 60)
    logger.info("ROCm training completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
