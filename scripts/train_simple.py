#!/usr/bin/env python3
"""
Hungarian Whisper Training - Simple Direct Training
No LoRA, just verify training loop works
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np

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


class SimpleDataset:
    """Simple dataset for quick training."""

    def __init__(self, num_samples=200, feature_extractor=None, tokenizer=None):
        self.num_samples = num_samples
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

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

        if self.feature_extractor:
            input_features = self.feature_extractor(
                audio,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features[0]
        else:
            input_features = torch.randn(80, 300)

        if self.tokenizer:
            labels = self.tokenizer(
                transcription,
                return_tensors="pt"
            ).input_ids[0]
        else:
            labels = torch.randint(0, 100, (20,))

        return (input_features, labels)


def main():
    logger.info("=" * 60)
    logger.info("Hungarian Whisper Simple Training")
    logger.info("=" * 60)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if not torch.cuda.is_available():
        logger.error("ROCm not available!")
        sys.exit(1)

    device = torch.device("cuda")
    allocated = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"VRAM: {allocated:.2f}GB / {total:.2f}GB")

    # Load model directly without LoRA
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    logger.info("\n[1] Loading Whisper model...")
    model_name = "openai/whisper-small"

    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    logger.info(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

    model = model.to(device)
    model.train()

    allocated = torch.cuda.memory_allocated() / 1e9
    logger.info(f"VRAM after model load: {allocated:.2f}GB")

    # Create dataset
    logger.info("\n[2] Creating dataset...")
    train_dataset = SimpleDataset(200, processor.feature_extractor, processor.tokenizer)

    # Training
    logger.info("\n[3] Starting training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    batch_size = 2
    max_steps = 30

    step = 0
    for epoch in range(1):
        for i in range(0, len(train_dataset), batch_size):
            if step >= max_steps:
                break

            # Get batch
            batch_features = []
            batch_labels = []
            for j in range(i, min(i + batch_size, len(train_dataset))):
                feat, lab = train_dataset[j]
                batch_features.append(feat)
                batch_labels.append(lab)

            # Pad sequences
            batch_features = torch.nn.utils.rnn.pad_sequence(batch_features, batch_first=True)
            batch_labels = torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True)

            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            # Forward
            outputs = model(input_features=batch_features, labels=batch_labels)
            loss = outputs.loss

            # Backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % 5 == 0:
                allocated = torch.cuda.memory_allocated() / 1e9
                logger.info(f"Step {step}/{max_steps} | Loss: {loss.item():.4f} | VRAM: {allocated:.2f}GB")

            step += 1

        if step >= max_steps:
            break

    # Save
    allocated = torch.cuda.memory_allocated() / 1e9
    logger.info(f"\nFinal VRAM: {allocated:.2f}GB")

    logger.info("\n[4] Saving model...")
    output_dir = "./output/checkpoints_simple"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

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
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("\n" + "=" * 60)
    logger.info("Training completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()