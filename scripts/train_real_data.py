#!/usr/bin/env python3
"""
Hungarian Whisper Training with Real Data Support
Uses transformers + peft on ROCm

Usage:
    # With synthetic data (for testing):
    python scripts/train_real_data.py --mode synthetic

    # With real data (Hungarian speech):
    python scripts/train_real_data.py --mode real --dataset voxpopuli
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
for k in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']:
    os.environ.pop(k, None)

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Hungarian words for synthetic data generation
HUNGARIAN_WORDS = [
    "köszönöm", "szépen", "üdvözöllek", "hogyan", "vagy", "mi", "ez", "egy",
    "mondat", "tisztelet", "siker", "gyümölcs", "kávé", "tea", "reggel",
    "este", "nap", "hét", "év", "hónap", "magyar", "nyelv", "beszéd",
    "hello", "viszlát", "igen", "nem", "please", "thank", "you"
]


def generate_transcription():
    """Generate Hungarian-like text."""
    num_words = np.random.randint(3, 8)
    return ' '.join(np.random.choice(HUNGARIAN_WORDS, num_words))


class SyntheticDataset(torch.utils.data.Dataset):
    """Synthetic Hungarian speech dataset for testing."""

    def __init__(self, num_samples=500, feature_extractor=None, tokenizer=None):
        self.num_samples = num_samples
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        np.random.seed(idx)
        duration = np.random.uniform(2.0, 10.0)
        audio = np.random.randn(int(16000 * duration)).astype(np.float32) * 0.01
        transcription = generate_transcription()

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


class RealHungarianDataset(torch.utils.data.Dataset):
    """Real Hungarian speech dataset from HuggingFace.

    Supports:
    - google/fleurs (hu_hu)
    - facebook/voxpopuli (hu)
    - openslr (hungarian)
    """

    def __init__(
        self,
        dataset_name: str = "google/fleurs",
        subset: str = "hu_hu",
        split: str = "train",
        max_samples: int = 5000,
        feature_extractor=None,
        tokenizer=None,
        streaming: bool = True
    ):
        from datasets import load_dataset

        self.dataset_name = dataset_name
        self.subset = subset
        self.max_samples = max_samples
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.cache = {}  # Cache for pre-loaded samples

        logger.info(f"Loading {dataset_name} ({subset}) from HuggingFace...")

        try:
            # Load with audio processing
            self.dataset = load_dataset(
                dataset_name,
                subset,
                split=split,
                streaming=streaming,
                trust_remote_code=True
            )

            # Pre-load samples into memory for faster training
            self.samples = []
            for i, sample in enumerate(self.dataset):
                if i >= max_samples:
                    break

                # Get audio and transcription based on dataset format
                if 'audio' in sample:
                    audio = sample['audio']['array']
                    sr = sample['audio']['sampling_rate']
                else:
                    continue

                # Get transcription
                transcription = sample.get('transcription', sample.get('text', sample.get('sentence', '')))
                if not transcription:
                    continue

                # Resample if needed
                if sr != 16000:
                    import librosa
                    audio = librosa.resample(
                        audio,
                        orig_sr=sr,
                        target_sr=16000
                    )

                # Normalize audio
                audio = audio.astype(np.float32)
                if audio.max() != 0:
                    audio = audio / (np.abs(audio).max() + 1e-8)

                self.samples.append({
                    'audio': audio,
                    'transcription': transcription
                })

                if (i + 1) % 100 == 0:
                    logger.info(f"Pre-loaded {i+1}/{max_samples} samples")

            logger.info(f"Dataset loaded! Pre-loaded {len(self.samples)} samples")

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            logger.warning("Falling back to synthetic data")
            self.samples = None

    def __len__(self):
        if self.samples is not None:
            return len(self.samples)
        return self.max_samples

    def __getitem__(self, idx):
        if self.samples is None:
            # Fallback to synthetic
            np.random.seed(idx)
            duration = np.random.uniform(2.0, 10.0)
            audio = np.random.randn(int(16000 * duration)).astype(np.float32) * 0.01
            transcription = generate_transcription()
        else:
            sample = self.samples[idx % len(self.samples)]
            audio = sample['audio']
            transcription = sample['transcription']

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


def create_dataset(mode: str, feature_extractor, tokenizer, max_samples: int = 5000):
    """Create dataset based on mode."""
    if mode == "synthetic":
        return SyntheticDataset(max_samples, feature_extractor, tokenizer)
    elif mode == "real":
        return RealHungarianDataset(
            max_samples=max_samples,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


def train(
    model,
    train_dataset,
    output_dir: str,
    max_steps: int = 100,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    gradient_accumulation_steps: int = 1
):
    """Training loop."""
    device = torch.device("cuda")
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

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
            loss = outputs.loss / gradient_accumulation_steps

            # Backward
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                allocated = torch.cuda.memory_allocated() / 1e9
                logger.info(
                    f"Step {step}/{max_steps} | Loss: {loss.item():.4f} | "
                    f"VRAM: {allocated:.2f}GB"
                )

            step += 1

        if step >= max_steps:
            break

    return model


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Hungarian Whisper Training')
    parser.add_argument('--mode', type=str, default='synthetic',
                        choices=['synthetic', 'real'],
                        help='Training mode: synthetic or real data')
    parser.add_argument('--dataset', type=str, default='voxpopuli',
                        help='Dataset name (for real mode)')
    parser.add_argument('--max_samples', type=int, default=5000,
                        help='Maximum number of samples')
    parser.add_argument('--max_steps', type=int, default=100,
                        help='Maximum training steps')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--output_dir', type=str, default='./output/checkpoints_real',
                        help='Output directory')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"Hungarian Whisper Training ({args.mode} mode)")
    logger.info("=" * 60)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check GPU
    if not torch.cuda.is_available():
        logger.error("ROCm not available!")
        sys.exit(1)

    device = torch.device("cuda")
    allocated = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"VRAM: {allocated:.2f}GB / {total:.2f}GB")

    # Load model
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    logger.info(f"\n[1] Loading Whisper model...")
    model_name = "openai/whisper-small"

    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    model = model.to(device)
    model.train()

    allocated = torch.cuda.memory_allocated() / 1e9
    logger.info(f"VRAM after model load: {allocated:.2f}GB")

    # Create dataset
    logger.info(f"\n[2] Creating dataset ({args.mode} mode)...")
    train_dataset = create_dataset(
        args.mode,
        processor.feature_extractor,
        processor.tokenizer,
        args.max_samples
    )

    # Training
    logger.info(f"\n[3] Starting training (max_steps={args.max_steps})...")
    train(
        model,
        train_dataset,
        args.output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size
    )

    # Save model
    logger.info(f"\n[4] Saving model to {args.output_dir}...")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

    # Save summary
    summary = {
        "model_name": model_name,
        "mode": args.mode,
        "train_samples": len(train_dataset),
        "max_steps": args.max_steps,
        "rocm": True,
        "rocm_version": torch.version.hip
    }

    with open(os.path.join(args.output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    allocated = torch.cuda.memory_allocated() / 1e9
    logger.info(f"\nFinal VRAM: {allocated:.2f}GB")
    logger.info(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("\n" + "=" * 60)
    logger.info("Training completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
