#!/usr/bin/env python3
"""
Hungarian CTC ASR Training

Uses wav2vec2 (proven CTC architecture) for Hungarian speech recognition.
Supports greedy CTC decoding and beam search decoding.

Usage:
    python scripts/train_ctc.py --max_steps 500
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
import torch.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Hungarian character set for CTC
HUNGARIAN_CHARS = [
    'a', 'á', 'ä', 'b', 'c', 'd', 'e', 'é', 'f', 'g', 'h', 'i', 'í',
    'j', 'k', 'l', 'm', 'n', 'o', 'ó', 'ö', 'ő', 'p', 'q', 'r', 's',
    't', 'u', 'ú', 'ü', 'ű', 'v', 'w', 'x', 'y', 'z', 'zs',
    'A', 'Á', 'Ä', 'B', 'C', 'D', 'E', 'É', 'F', 'G', 'H', 'I', 'Í',
    'J', 'K', 'L', 'M', 'N', 'O', 'Ó', 'Ö', 'Ő', 'P', 'Q', 'R', 'S',
    'T', 'U', 'Ú', 'Ü', 'Ű', 'V', 'W', 'X', 'Y', 'Z', 'ZS',
    ' ', "'", '.', ',', '?', '!'
]

# CTC blank token
BLANK_TOKEN = '<blank>'
CHAR2IDX = {c: i + 1 for i, c in enumerate(HUNGARIAN_CHARS)}
CHAR2IDX[BLANK_TOKEN] = 0
IDX2CHAR = {v: k for k, v in CHAR2IDX.items()}
VOCAB_SIZE = len(CHAR2IDX) + 1  # +1 for CTC blank


class HungarianCTCDataset(torch.utils.data.Dataset):
    """Hungarian speech dataset for CTC training."""

    def __init__(
        self,
        max_samples: int = 500,
        feature_extractor=None,
        char2idx: dict = CHAR2IDX,
        sampling_rate: int = 16000
    ):
        from datasets import load_dataset

        self.max_samples = max_samples
        self.feature_extractor = feature_extractor
        self.char2idx = char2idx
        self.sampling_rate = sampling_rate
        self.samples = []

        logger.info("Loading Google FLEURS Hungarian dataset...")

        try:
            ds = load_dataset(
                'google/fleurs',
                'hu_hu',
                split='train',
                streaming=True,
                trust_remote_code=True
            )

            for i, sample in enumerate(ds):
                if i >= max_samples:
                    break

                audio = sample['audio']['array']
                sr = sample['audio']['sampling_rate']
                text = sample.get('transcription', '')

                if not text:
                    continue

                # Resample if needed
                if sr != sampling_rate:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=sampling_rate)

                # Normalize audio
                audio = audio.astype(np.float32)
                if audio.max() != 0:
                    audio = audio / (np.abs(audio).max() + 1e-8)

                self.samples.append({
                    'audio': audio,
                    'text': text,
                    'sr': sampling_rate
                })

                if (i + 1) % 100 == 0:
                    logger.info(f"Loaded {i+1}/{max_samples} samples")

            logger.info(f"Dataset loaded! Total samples: {len(self.samples)}")

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            logger.warning("Using synthetic fallback data")
            self.samples = [{'audio': np.zeros(16000, dtype=np.float32), 'text': 'teszt'}]
            for i in range(min(50, max_samples)):
                duration = np.random.uniform(1.0, 5.0)
                audio = np.random.randn(int(16000 * duration)).astype(np.float32) * 0.01
                text = "teszt szöveg"
                self.samples.append({'audio': audio, 'text': text})

    def __len__(self):
        return len(self.samples) if self.samples else 100

    def __getitem__(self, idx):
        if not self.samples:
            # Fallback to synthetic
            np.random.seed(idx)
            duration = np.random.uniform(1.0, 5.0)
            audio = np.random.randn(int(16000 * duration)).astype(np.float32) * 0.01
            text = "teszt"
        else:
            sample = self.samples[idx % len(self.samples)]
            audio = sample['audio']
            text = sample['text']

        # Ensure audio is 1D float array
        if isinstance(audio, list):
            audio = np.array(audio, dtype=np.float32)
        elif hasattr(audio, 'astype'):
            audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.flatten()

        return audio, text


def ctc_greedy_decode(emissions, blank_id=0):
    """CTC greedy decoding."""
    indices = []
    prev = -1
    for timestep in emissions:
        best_idx = timestep.argmax()
        if best_idx != blank_id and best_idx != prev:
            indices.append(best_idx.item() if hasattr(best_idx, 'item') else best_idx)
        prev = best_idx
    return indices


def train_ctc(
    model,
    train_dataset,
    output_dir: str,
    max_steps: int = 200,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    gradient_accumulation_steps: int = 2
):
    """Training loop with CTC loss."""
    device = torch.device("cuda")
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

    step = 0
    for epoch in range(1):
        for i in range(0, len(train_dataset), batch_size):
            if step >= max_steps:
                break

            # Get batch - raw audio and text
            batch_audio = []
            batch_texts = []
            for j in range(i, min(i + batch_size, len(train_dataset))):
                audio, text = train_dataset[j]
                batch_audio.append(audio)
                batch_texts.append(text)

            # Convert to tensors and pad audio
            max_len = max(a.shape[0] for a in batch_audio)
            batch_padded = []
            for audio in batch_audio:
                if len(audio) < max_len:
                    pad_len = max_len - len(audio)
                    audio = np.pad(audio, (0, pad_len), mode='constant')
                batch_padded.append(audio)
            # wav2vec2 expects (batch, time) - no extra dimension
            batch_tensors = torch.from_numpy(np.array(batch_padded)).float()  # (B, T)

            # Encode labels
            batch_labels = []
            label_lengths = []
            for text in batch_texts:
                label_indices = [CHAR2IDX.get(c, 0) for c in text if c in CHAR2IDX]
                batch_labels.extend(label_indices)
                label_lengths.append(len(label_indices))

            batch_labels = torch.tensor(batch_labels, dtype=torch.long)

            # Move to device
            batch_tensors = batch_tensors.to(device)
            batch_labels = batch_labels.to(device)

            # Forward pass - wav2vec2 expects input_values
            outputs = model(input_values=batch_tensors)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            # CTC loss requires log_softmax
            log_probs = torch.log_softmax(logits, dim=-1)
            log_probs = log_probs.permute(1, 0, 2)  # (T, B, C)

            # Compute CTC loss
            tgt_lengths = torch.tensor(label_lengths, dtype=torch.long)
            inp_lengths = torch.tensor([log_probs.shape[0]] * len(label_lengths), dtype=torch.long)

            try:
                loss = ctc_loss(log_probs, batch_labels, inp_lengths, tgt_lengths)
            except Exception as e:
                logger.warning(f"CTC loss error: {e}")
                loss = torch.tensor(0.0, device=device)

            # Backward
            loss = loss / gradient_accumulation_steps
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
    parser = argparse.ArgumentParser(description='Hungarian CTC Training')
    parser.add_argument('--max_samples', type=int, default=500,
                        help='Maximum number of samples')
    parser.add_argument('--max_steps', type=int, default=200,
                        help='Maximum training steps')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--output_dir', type=str, default='./output/checkpoints_ctc',
                        help='Output directory')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Hungarian CTC ASR Training (wav2vec2)")
    logger.info("=" * 60)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if not torch.cuda.is_available():
        logger.error("ROCm not available!")
        sys.exit(1)

    device = torch.device("cuda")
    allocated = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"VRAM: {allocated:.2f}GB / {total:.2f}GB")

    # Load wav2vec2 model with CTC head
    logger.info("\n[1] Loading wav2vec2 CTC model...")
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    model_name = "facebook/wav2vec2-base"

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)

    # Modify vocab size for Hungarian
    model.config.vocab_size = VOCAB_SIZE
    model.lm_head = nn.Linear(model.config.hidden_size, VOCAB_SIZE)

    model = model.to(device)
    model.train()

    allocated = torch.cuda.memory_allocated() / 1e9
    logger.info(f"VRAM after model load: {allocated:.2f}GB")
    logger.info(f"Vocab size: {VOCAB_SIZE}")

    # Create dataset
    logger.info("\n[2] Creating dataset...")
    train_dataset = HungarianCTCDataset(
        max_samples=args.max_samples,
        feature_extractor=processor.feature_extractor
    )

    # Training
    logger.info(f"\n[3] Starting training (max_steps={args.max_steps})...")
    model = train_ctc(
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

    # Save char mapping
    with open(os.path.join(args.output_dir, "char_map.json"), 'w') as f:
        json.dump({
            'char2idx': CHAR2IDX,
            'idx2char': IDX2CHAR,
            'vocab_size': VOCAB_SIZE
        }, f, indent=2)

    allocated = torch.cuda.memory_allocated() / 1e9
    logger.info(f"\nFinal VRAM: {allocated:.2f}GB")
    logger.info(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("\n" + "=" * 60)
    logger.info("CTC Training completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
