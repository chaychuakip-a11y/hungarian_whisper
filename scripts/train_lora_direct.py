#!/usr/bin/env python3
"""
Hungarian Whisper Training - Direct PyTorch Training Loop
Bypasses peft to avoid compatibility issues
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass

from data.hungarian_normalizer import HungarianTextNormalizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_transcription():
    """Generate Hungarian-like text."""
    words = ["köszönöm", "szépen", "üdvözöllek", "hogyan", "vagy", "mi", "ez", "egy",
             "mondat", "tisztelet", "siker", "gyümölcs", "kávé", "tea", "reggel",
             "este", "nap", "hét", "év", "hónap", "magyar", "nyelv", "beszéd"]
    num_words = np.random.randint(3, 8)
    return ' '.join(np.random.choice(words, num_words))


class LoRALinear(nn.Module):
    """Simple LoRA implementation for linear layers."""

    def __init__(self, linear_layer, rank=16, alpha=128, dropout=0.05):
        super().__init__()
        self.linear = linear_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Freeze original weights
        for param in self.linear.parameters():
            param.requires_grad = False

        # LoRA parameters
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features

        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # Original forward pass (frozen)
        original = self.linear(x)

        # LoRA forward pass
        lora = self.dropout(x) @ self.lora_A.T @ self.lora_B.T

        return original + lora * self.scaling


def apply_lora_to_model(model, rank=16, alpha=128, dropout=0.05):
    """Apply LoRA to Whisper model attention layers."""
    # Target modules: q_proj and v_proj in attention layers
    for name, module in model.named_modules():
        if 'q_proj' in name or 'v_proj' in name:
            # Get parent module and attribute name
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]

            parent = model.get_submodule(parent_name) if parent_name else model

            if hasattr(parent, attr_name):
                original = getattr(parent, attr_name)
                if isinstance(original, nn.Linear):
                    lora_linear = LoRALinear(original, rank, alpha, dropout)
                    setattr(parent, attr_name, lora_linear)
                    logger.info(f"Applied LoRA to {name}")

    return model


@dataclass
class SimpleSpeechSample:
    """Simple speech sample for training."""
    input_features: torch.Tensor
    labels: torch.Tensor


class SimpleDataset:
    """Simple dataset for quick training."""

    def __init__(self, num_samples=500, feature_extractor=None, tokenizer=None):
        self.num_samples = num_samples
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


def count_trainable_parameters(model):
    """Count trainable parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def main():
    logger.info("=" * 60)
    logger.info("Hungarian Whisper Training (Direct LoRA)")
    logger.info("=" * 60)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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

    # Apply custom LoRA
    logger.info("\n[1b] Applying custom LoRA...")
    model = apply_lora_to_model(model, rank=16, alpha=128, dropout=0.05)

    trainable, total = count_trainable_parameters(model)
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

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

    # Training
    logger.info("\n[3] Starting training...")
    model.train()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )

    batch_size = 4
    num_epochs = 1
    max_steps = 50

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
    logger.info("\n[4] Saving model...")
    output_dir = "./output/checkpoints_lora_direct"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    # Save summary
    summary = {
        "model_name": model_name,
        "train_samples": len(train_dataset),
        "max_steps": max_steps,
        "final_loss": float(loss.item()),
        "trainable_params": trainable,
        "total_params": total,
        "rocm": True,
        "lora_type": "custom"
    }

    with open(os.path.join(output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nModel saved to {output_dir}")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("\n" + "=" * 60)
    logger.info("Direct LoRA training completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()