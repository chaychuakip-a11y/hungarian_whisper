"""
Synthetic Hungarian ASR data generator for testing the pipeline.
Generates fake audio features and transcriptions for testing.
"""

import json
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import struct

# Hungarian character set for generating realistic-looking text
HUNGARIAN_CHARS = "aábcdeéfghiíjklmnoóöőpqrstuúüűvwxyzAÁBCDEÉFGHIÍJKLMNOÓÖŐPQRSTUÚÜŰVWXYZ .,!?"
SAMPLE_HUNGARIAN_WORDS = [
    "köszönöm", "szépen", "üdvözöllek", "hogyan", " vagy", "mi", "ez", "egy",
    "mondat", "首都", "tisztelet", "siker", "gyümölcs", "kávé", "tea",
    "reggel", "este", "nap", "hét", "év", "hónap", "magyar", "nyelv",
    "beszéd", "felismerés", "gép", "tanulás", "mély", "neurális", "hálózat"
]


def generate_synthetic_audio(duration_secs: float = 5.0, sr: int = 16000) -> np.ndarray:
    """Generate synthetic audio (silence with some noise)."""
    num_samples = int(duration_secs * sr)
    # Generate silence with low-level noise
    audio = np.random.randn(num_samples).astype(np.float32) * 0.001
    # Add some random sine waves to make it more realistic
    for _ in range(3):
        freq = np.random.randint(100, 500)
        phase = np.random.random() * 2 * np.pi
        t = np.linspace(0, duration_secs, num_samples)
        audio += 0.01 * np.sin(2 * np.pi * freq * t + phase)
    return audio


def generate_synthetic_transcription(min_words: int = 3, max_words: int = 10) -> str:
    """Generate synthetic Hungarian-like text."""
    num_words = np.random.randint(min_words, max_words + 1)
    words = []
    for _ in range(num_words):
        word_len = np.random.randint(2, 8)
        word = ''.join(np.random.choice(list(HUNGARIAN_CHARS), word_len))
        words.append(word)
    return ' '.join(words)


def create_synthetic_dataset(
    num_samples: int,
    min_duration: float = 1.0,
    max_duration: float = 25.0,
    output_dir: str = "./data/synthetic"
) -> Tuple[List[Dict], Path]:
    """Create synthetic Hungarian ASR dataset for testing.

    Returns:
        List of sample dicts and output directory path.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    samples = []
    audio_dir = output_path / "audio"
    audio_dir.mkdir(exist_ok=True)

    print(f"Generating {num_samples} synthetic samples...")

    for i in range(num_samples):
        duration = np.random.uniform(min_duration, max_duration)
        audio = generate_synthetic_audio(duration)

        # Save audio as numpy file (simpler than wav for testing)
        audio_path = audio_dir / f"sample_{i:06d}.npy"
        np.save(audio_path, audio)

        transcription = generate_synthetic_transcription()

        samples.append({
            "id": f"sample_{i:06d}",
            "audio_path": str(audio_path),
            "audio": {
                "path": str(audio_path),
                "array": audio,
                "sampling_rate": 16000
            },
            "text": transcription,
            "normalized_text": transcription.lower(),
            "duration": duration
        })

        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples")

    return samples, output_path


def save_synthetic_dataset_info(
    samples: List[Dict],
    output_path: Path,
    train_ratio: float = 0.9
) -> Dict:
    """Save dataset info JSON file."""
    num_samples = len(samples)
    split_idx = int(num_samples * train_ratio)

    dataset_info = {
        "num_samples": num_samples,
        "train_samples": split_idx,
        "eval_samples": num_samples - split_idx,
        "audio_dir": str(output_path / "audio"),
        "train_ratio": train_ratio,
        "min_duration": 1.0,
        "max_duration": 25.0
    }

    info_path = output_path / "dataset_info.json"
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)

    return dataset_info


class SyntheticHungarianDataset:
    """Synthetic Hungarian dataset for pipeline testing."""

    def __init__(self, data_dir: str = "./data/synthetic"):
        self.data_dir = Path(data_dir)
        info_path = self.data_dir / "dataset_info.json"

        if not info_path.exists():
            raise FileNotFoundError(f"Dataset not found at {data_dir}. Run with --generate first.")

        with open(info_path) as f:
            self.info = json.load(f)

        self.audio_dir = Path(self.info["audio_dir"])
        self.num_samples = self.info["num_samples"]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        audio_path = self.audio_dir / f"sample_{idx:06d}.npy"
        audio = np.load(audio_path)

        # Generate a transcription on the fly (since we didn't save transcriptions)
        transcription = generate_synthetic_transcription()

        return {
            "id": f"sample_{idx:06d}",
            "audio": {
                "path": str(audio_path),
                "array": audio,
                "sampling_rate": 16000
            },
            "text": transcription,
            "normalized_text": transcription.lower(),
            "duration": len(audio) / 16000
        }

    def split(self, train_ratio: float = 0.9):
        """Split into train and eval datasets."""
        split_idx = int(self.num_samples * train_ratio)

        class SubDataset:
            def __init__(self, parent, start, end):
                self.parent = parent
                self.start = start
                self.end = end

            def __len__(self):
                return self.end - self.start

            def __getitem__(self, idx):
                return self.parent[idx + self.start]

        return SubDataset(self, 0, split_idx), SubDataset(self, split_idx, self.num_samples)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="./data/synthetic")
    args = parser.parse_args()

    samples, output_path = create_synthetic_dataset(
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )

    save_synthetic_dataset_info(samples, output_path)

    print(f"\nSynthetic dataset created at: {output_path}")
    print(f"Total samples: {len(samples)}")
