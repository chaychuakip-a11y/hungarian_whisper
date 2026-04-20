#!/usr/bin/env python3
"""
Hungarian ASR Data Preparation for Hulk Framework

This script prepares Hungarian speech data in the LMDB format required by Hulk framework.
Based on /home/lty/am/whisper_ctc/examples/whisper_ll/data_preprocess/multilingual_data_prepare.py

Usage:
    python scripts/prepare_hulk_data.py --output_dir ./data/hulk_lmdb
"""

import argparse
import json
import logging
import os
import struct
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

# Add Hulk to path
HULK_DIR = "/home/lty/am/whisper_ctc"
sys.path.insert(0, os.path.join(HULK_DIR))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


# Hungarian phone set
HUNGARIAN_PHONES = [
    '<SIL>', '<SP>',
    'a', 'a:', 'e', 'e:', 'i', 'i:', 'o', 'o:', 'o:', 'u', 'u:', 'u:',
    'y', 'y:', 'o:', 'o:', 'o:',
    'b', 'c', 'd', 'dz', 'dzs', 'f', 'g', 'gy', 'h', 'j', 'k', 'l', 'ly',
    'm', 'n', 'ny', 'p', 'q', 'r', 's', 'sz', 't', 'ty', 'v', 'w', 'x', 'y', 'z', 'zs'
]

# Phone to index (0 is blank for CTC)
PHONE2IDX = {p: i + 1 for i, p in enumerate(HUNGARIAN_PHONES)}
PHONE2IDX['<blank>'] = 0


class ChunkSeqGenerator:
    """Generate chunk index file for Hulk training.

    Based on whisper_ctc multilingual_data_prepare.py IndexChunkGenerator
    """

    def __init__(self, chunk_size: int = 10000, seed: int = 0):
        self.chunk_size = chunk_size
        self.seed = seed
        self._lang_indices = {}
        self._chunks = []

    def generate(
        self,
        samples: List[Dict],
        save_path: str,
        lang: str = "hungarian"
    ) -> None:
        """Generate chunk index file.

        Args:
            samples: List of sample dicts with 'id', 'audio_path', 'text'
            save_path: Output path for chunk .bin file
            lang: Language identifier
        """
        import random
        random.seed(self.seed)

        # Create sample indices
        lang_indices = list(range(len(samples)))
        random.shuffle(lang_indices)
        self._lang_indices[lang] = lang_indices

        # Generate chunks
        chunks = []
        for i in range(0, len(lang_indices), self.chunk_size):
            chunk = lang_indices[i:i + self.chunk_size]
            chunks.append(chunk)

        # Write chunk file
        self._write_chunk_file(chunks, save_path)
        logger.info(f"Generated {len(chunks)} chunks, saved to {save_path}")

    def _write_chunk_file(self, chunks: List[List[int]], save_path: str) -> None:
        """Write chunk index file in Hulk format."""
        with open(save_path, 'wb') as f:
            # Header: num_chunks (4 bytes), max_chunk_size (4 bytes), md5 (16 bytes)
            num_chunks = len(chunks)
            max_chunk_size = max(len(c) for c in chunks) if chunks else 0

            # Write header
            f.write(struct.pack('<2I', num_chunks, max_chunk_size))

            # MD5 placeholder (16 bytes)
            f.write(b'\x00' * 16)

            # Write chunks
            for chunk in chunks:
                for idx in chunk:
                    f.write(struct.pack('<Q', idx))

                # Pad to max_chunk_size
                for _ in range(max_chunk_size - len(chunk)):
                    f.write(struct.pack('<Q', 0))


def text_to_phones(text: str) -> List[str]:
    """Convert text to phone sequence.

    Simplified version - in production would use g2p (grapheme-to-phoneme).
    """
    # For now, just split by whitespace
    return text.lower().split()


def create_dataset_json(
    samples: List[Dict],
    output_path: str,
    language: str = "hungarian"
) -> None:
    """Create dataset JSON file for Hulk.

    Format based on multilingual_data_prepare.py
    """
    dataset_specs = []

    for i, sample in enumerate(samples):
        spec = {
            "id": sample.get("id", f"sample_{i:06d}"),
            "audio_path": sample["audio_path"],
            "text": sample["text"],
            "duration": sample.get("duration", 0),
            "language": language
        }
        dataset_specs.append(spec)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_specs, f, ensure_ascii=False, indent=2)

    logger.info(f"Created dataset JSON with {len(samples)} samples: {output_path}")


def prepare_lmdb_data(
    samples: List[Dict],
    output_dir: str,
    language: str = "hungarian"
) -> Tuple[str, str, str]:
    """Prepare LMDB database for Hulk training.

    Args:
        samples: List of sample dicts
        output_dir: Output directory
        language: Language identifier

    Returns:
        Tuple of (lmdb_path, chunk_path, json_path)
    """
    import lmdb

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    lmdb_path = str(output_path / f"{language}_lmdb")
    chunk_path = str(output_path / f"{language}_chunk10000.bin")
    json_path = str(output_path / f"{language}_dataset.json")

    # Create dataset JSON
    create_dataset_json(samples, json_path, language)

    # Create LMDB database
    map_size = 500 * 1024 ** 3  # 500GB
    env = lmdb.open(lmdb_path, map_size=map_size, subdir=False, readonly=False)

    key_fmt = "<Q"
    with env.begin(write=True) as txn:
        for i, sample in enumerate(samples):
            key = struct.pack(key_fmt, i)

            # Prepare phone labels
            phones = text_to_phones(sample.get("text", ""))
            phone_indices = [PHONE2IDX.get(p, 0) for p in phones]

            value = json.dumps({
                "id": sample.get("id", f"sample_{i:06d}"),
                "audio_path": sample["audio_path"],
                "text": sample["text"],
                "phones": phones,
                "phone_indices": phone_indices,
                "language": language
            }, ensure_ascii=False).encode('utf-8')

            txn.put(key, value)

    logger.info(f"Created LMDB with {len(samples)} samples: {lmdb_path}")

    # Generate chunk indices
    generator = ChunkSeqGenerator(chunk_size=10000, seed=42)
    generator.generate(samples, chunk_path, language)

    return lmdb_path, chunk_path, json_path


def generate_synthetic_samples(num_samples: int = 1000) -> List[Dict]:
    """Generate synthetic Hungarian samples for testing."""
    import random

    HUNGARIAN_WORDS = [
        "köszönöm", "szépen", "üdvözöllek", "hogyan", "vagy", "mi", "ez", "egy",
        "mondat", "tisztelet", "siker", "gyümölcs", "kávé", "tea", "reggel",
        "este", "nap", "hét", "év", "hónap", "magyar", "nyelv", "beszéd"
    ]

    samples = []
    for i in range(num_samples):
        num_words = random.randint(3, 8)
        text = ' '.join(random.choice(HUNGARIAN_WORDS) for _ in range(num_words))

        samples.append({
            "id": f"sample_{i:06d}",
            "audio_path": f"/dummy/audio_{i:06d}.wav",
            "text": text,
            "duration": random.uniform(2.0, 15.0)
        })

    return samples


def main():
    parser = argparse.ArgumentParser(description='Prepare Hungarian data for Hulk')
    parser.add_argument('--output_dir', type=str, default='./data/hulk_lmdb',
                        help='Output directory')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of synthetic samples to generate')
    parser.add_argument('--generate_synthetic', action='store_true',
                        help='Generate synthetic data for testing')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Hungarian ASR Data Preparation (Hulk Framework)")
    logger.info("=" * 60)

    # Generate or load samples
    if args.generate_synthetic:
        logger.info(f"Generating {args.num_samples} synthetic samples...")
        samples = generate_synthetic_samples(args.num_samples)
    else:
        logger.info("No data provided. Use --generate_synthetic for testing.")
        logger.info("In production, load from HuggingFace or other sources.")
        return

    # Prepare LMDB data
    logger.info("\nPreparing LMDB data...")
    lmdb_path, chunk_path, json_path = prepare_lmdb_data(
        samples,
        args.output_dir,
        language="hungarian"
    )

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Data preparation complete!")
    logger.info("=" * 60)
    logger.info(f"LMDB path: {lmdb_path}")
    logger.info(f"Chunk path: {chunk_path}")
    logger.info(f"JSON path: {json_path}")
    logger.info(f"Total samples: {len(samples)}")

    # Save phone dictionary
    phone_dict_path = Path(args.output_dir) / "phone_dict.json"
    with open(phone_dict_path, 'w') as f:
        json.dump(PHONE2IDX, f, indent=2)
    logger.info(f"Phone dict: {phone_dict_path}")

    logger.info("\nTo train with Hulk framework, update the YAML config with these paths.")


if __name__ == "__main__":
    main()
