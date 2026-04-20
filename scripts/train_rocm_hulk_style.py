#!/usr/bin/env python3
"""
Hungarian Whisper Training on ROCm (Hulk-style simulation)

This script simulates a Hulk framework training run on local ROCm hardware.
It uses the same data format and training patterns as the Hulk framework.
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
             "mondat", "tisztelet", "siker", "gyümölcs", "kávé", "tea", "reggel",
             "este", "nap", "hét", "év", "hónap", "magyar", "nyelv", "beszéd"]
    num_words = np.random.randint(3, 8)
    return ' '.join(np.random.choice(words, num_words))


@dataclass
class SimpleSpeechSample:
    """Simple speech sample for training."""
    input_features: torch.Tensor
    labels: torch.Tensor


class SimpleDataset:
    """Simple dataset for Hulk-style LMDB data loading."""

    def __init__(self, num_samples=1000, feature_extractor=None, tokenizer=None):
        self.num_samples = num_samples
        self.normalizer = HungarianTextNormalizer()
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

        # Simulate LMDB data loading (chunk-based)
        self.chunk_size = 10000
        self.num_chunks = (num_samples + self.chunk_size - 1) // self.chunk_size

        # Pre-generate all samples
        logger.info(f"Loading {num_samples} samples from LMDB-like storage...")
        self.samples = []
        for i in range(num_samples):
            np.random.seed(i)
            duration = np.random.uniform(2.0, 10.0)
            audio = np.random.randn(int(16000 * duration)).astype(np.float32) * 0.01
            transcription = generate_transcription()

            self.samples.append({
                "id": f"sample_{i:06d}",
                "audio": audio,
                "transcription": transcription,
                "duration": duration
            })

        logger.info(f"Loaded {num_samples} samples in {self.num_chunks} chunks")

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


def train_step(model, batch_features, batch_labels, optimizer):
    """Single training step (Hulk-style)."""
    # Forward
    outputs = model(
        input_features=batch_features,
        labels=batch_labels
    )
    loss = outputs.loss

    # Backward
    loss.backward()

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()


def evaluate(model, eval_dataset, device, batch_size=8):
    """Run evaluation (Hulk-style validation)."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for i in range(0, min(len(eval_dataset), 50), batch_size):
            batch_features = []
            batch_labels = []
            for j in range(i, min(i + batch_size, len(eval_dataset))):
                sample = eval_dataset[j]
                batch_features.append(sample.input_features)
                batch_labels.append(sample.labels)

            batch_features = torch.nn.utils.rnn.pad_sequence(batch_features, batch_first=True)
            batch_labels = torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True)

            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(
                input_features=batch_features,
                labels=batch_labels
            )
            total_loss += outputs.loss.item()
            num_batches += 1

    model.train()
    return total_loss / num_batches if num_batches > 0 else 0


def main():
    logger.info("=" * 60)
    logger.info("Hungarian Whisper Training (Hulk-style ROCm)")
    logger.info("=" * 60)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if not torch.cuda.is_available():
        logger.error("ROCm not available!")
        sys.exit(1)

    # Check memory
    allocated = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"VRAM: {allocated:.2f}GB / {total:.2f}GB")

    # Load model (Hulk-style config)
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    logger.info("\n[1] Loading Whisper model...")
    model_name = "openai/whisper-small"

    # Use local cache
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--openai--whisper-small")
    if os.path.exists(cache_dir):
        logger.info(f"Using cached model from {cache_dir}")

    # Simulate Hulk config loading
    config = {
        "model": {
            "from_pretrained": model_name,
            "n_audio_ctx": 1500,
            "n_audio_state": 1280,
            "n_text_state": 1280,
            "n_audio_layer": 32,
            "n_text_layer": 4,
        },
        "lora": {
            "apply_lora": True,
            "lora_rank": 16,
            "lora_alpha": 128,
            "lora_dropout": 0.05,
        },
        "training": {
            "max_epoch": 2,
            "batch_size": 4,
            "lr": 5e-4,
            "clip_norm": 12.0,
        }
    }

    logger.info(f"Config: LoRA rank={config['lora']['lora_rank']}, "
                f"alpha={config['lora']['lora_alpha']}, "
                f"epochs={config['training']['max_epoch']}")

    # Load from local cache to avoid network issues
    cache_base = os.path.expanduser("~/.cache/huggingface/hub/models--openai--whisper-small")
    snapshot_dir = os.path.join(cache_base, "snapshots/973afd24965f72e36ca33b3055d56a652f456b4d")
    if os.path.exists(snapshot_dir):
        local_model_path = snapshot_dir
    else:
        local_model_path = model_name

    processor = WhisperProcessor.from_pretrained(local_model_path)
    model = WhisperForConditionalGeneration.from_pretrained(local_model_path)

    # Apply LoRA (Hulk-style)
    from peft import LoraConfig, get_peft_model, TaskType

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=config['lora']['lora_rank'],
        lora_alpha=config['lora']['lora_alpha'],
        target_modules=["q_proj", "v_proj"],
        lora_dropout=config['lora']['lora_dropout']
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Move to GPU
    device = torch.device("cuda")
    model = model.to(device)

    # Check memory after model load
    allocated = torch.cuda.memory_allocated() / 1e9
    logger.info(f"VRAM after model load: {allocated:.2f}GB")

    # Create dataset (Hulk-style LMDB loading)
    logger.info("\n[2] Creating dataset (simulating Hulk LMDB)...")
    num_train_samples = 2000
    num_eval_samples = 200

    train_dataset = SimpleDataset(num_train_samples, processor.feature_extractor, processor.tokenizer)
    eval_dataset = SimpleDataset(num_eval_samples, processor.feature_extractor, processor.tokenizer)

    logger.info(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

    # Export to HTK (Hulk-style)
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

    # Training (Hulk-style with distributed-like setup)
    logger.info("\n[4] Starting training...")
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'])

    # Hulk-style training parameters
    batch_size = config['training']['batch_size']
    num_epochs = config['training']['max_epoch']
    max_steps_per_epoch = 100
    total_steps = 0
    warmup_steps = 20

    for epoch in range(num_epochs):
        logger.info(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

        epoch_loss = 0
        for i in range(0, min(len(train_dataset), batch_size * max_steps_per_epoch), batch_size):
            step = i // batch_size
            if step >= max_steps_per_epoch:
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

            # Training step
            loss = train_step(model, batch_features, batch_labels, optimizer)
            epoch_loss += loss
            total_steps += 1

            # LR warmup (Hulk-style)
            if total_steps <= warmup_steps:
                lr = config['training']['lr'] * total_steps / warmup_steps
                for pg in optimizer.param_groups:
                    pg['lr'] = lr

            if step % 20 == 0:
                allocated = torch.cuda.memory_allocated() / 1e9
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Step {step}/{max_steps_per_epoch} | Loss: {loss:.4f} | "
                          f"LR: {current_lr:.2e} | VRAM: {allocated:.2f}GB")

        # Validation (Hulk-style)
        eval_loss = evaluate(model, eval_dataset, device)
        logger.info(f"Epoch {epoch + 1} | Train Loss: {epoch_loss / max_steps_per_epoch:.4f} | "
                   f"Eval Loss: {eval_loss:.4f}")

    # Final memory
    allocated = torch.cuda.memory_allocated() / 1e9
    logger.info(f"\nFinal VRAM: {allocated:.2f}GB")

    # Save model (Hulk-style checkpoint)
    logger.info("\n[5] Saving model...")
    output_dir = "./output/checkpoints_hulk_style"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    # Save summary (Hulk-style training log)
    summary = {
        "model_name": model_name,
        "train_samples": num_train_samples,
        "eval_samples": num_eval_samples,
        "total_steps": total_steps,
        "epochs": num_epochs,
        "lora_rank": config['lora']['lora_rank'],
        "lora_alpha": config['lora']['lora_alpha'],
        "batch_size": batch_size,
        "learning_rate": config['training']['lr'],
        "clip_norm": config['training']['clip_norm'],
        "final_loss": float(loss),
        "eval_loss": float(eval_loss),
        "rocm": True,
        "hulk_style": True,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(os.path.join(output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nModel saved to {output_dir}")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("\n" + "=" * 60)
    logger.info("Hulk-style ROCm training completed!")
    logger.info("=" * 60)

    return summary


if __name__ == "__main__":
    summary = main()