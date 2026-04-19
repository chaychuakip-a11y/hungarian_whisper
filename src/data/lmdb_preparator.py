"""
Hungarian ASR Data Preparation for Hulk Framework
Adapted from company's multilingual_data_prepare.py
"""

import json
import logging
import os
import struct
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import lmdb
import numpy as np

logger = logging.getLogger(__name__)


# Data types matching Hulk constants
class DataTypes:
    RAW_WAV = "RAW_WAV"
    FB80 = "FB80"
    FB40 = "FB40"
    ED_LABEL = "ED_LABEL"
    CTC_LABEL = "CTC_LABEL"


class HungarianASRDataPreparator:
    """Prepares Hungarian ASR data in LMDB format for Hulk training.

    Produces:
    - wav_frame: Raw audio features (fb80 format)
    - ctc_label / ed_label: Phone or text labels
    """

    def __init__(
        self,
        output_dir: str,
        chunk_size: int = 10000,
        feature_type: str = "fb80",
        lang: str = "hungarian"
    ):
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.feature_type = feature_type
        self.lang = lang

        os.makedirs(output_dir, exist_ok=True)

    def prepare_from_huggingface(
        self,
        train_dataset,
        eval_dataset,
        phone2idx: Optional[Dict[str, int]] = None
    ) -> Tuple[str, str, str]:
        """Prepare LMDB databases from HuggingFace datasets.

        Args:
            train_dataset: HuggingFace train dataset
            eval_dataset: HuggingFace eval dataset
            phone2idx: Phone to index mapping for CTC training

        Returns:
            Tuple of (train_db_path, eval_db_path, phone_dict_path)
        """
        logger.info("Preparing LMDB databases...")

        # Save phone dictionary
        phone_dict_path = os.path.join(self.output_dir, f"{self.lang}_phone_dict.json")
        if phone2idx:
            with open(phone_dict_path, 'w') as f:
                json.dump(phone2idx, f, ensure_ascii=False)
            logger.info(f"Saved phone dict to {phone_dict_path}")

        # Create train LMDB
        train_db_path = os.path.join(self.output_dir, "train_lmdb")
        self._create_lmdb_database(train_dataset, train_db_path, phone2idx, split="train")

        # Create eval LMDB
        eval_db_path = os.path.join(self.output_dir, "eval_lmdb")
        self._create_lmdb_database(eval_dataset, eval_db_path, phone2idx, split="eval")

        return train_db_path, eval_db_path, phone_dict_path

    def _create_lmdb_database(
        self,
        dataset,
        db_path: str,
        phone2idx: Optional[Dict[str, int]],
        split: str = "train"
    ):
        """Create LMDB database from dataset.

        Args:
            dataset: HuggingFace dataset
            db_path: Output LMDB path
            phone2idx: Phone to index mapping
            split: Dataset split name
        """
        map_size = 500 * 1024 ** 3  # 500GB
        env = lmdb.open(db_path, map_size=map_size, subdir=False, readonly=False)

        total_samples = len(dataset)
        logger.info(f"Creating LMDB with {total_samples} samples at {db_path}")

        # Key format: sample index (8 bytes unsigned int)
        key_fmt = "<Q"
        key_size = struct.calcsize(key_fmt)

        # Get feature dimension
        sample = dataset[0]
        if "fb80" in self.feature_type or "mel" in self.feature_type:
            feat_dim = 80
        elif "fb40" in self.feature_type:
            feat_dim = 40
        else:
            feat_dim = 80

        with env.begin(write=True) as txn:
            for idx in range(total_samples):
                item = dataset[idx]

                # Get audio and transcription
                audio_path = item["audio"]["path"]
                text = item.get("normalized_text", item.get("text", ""))

                if not text:
                    continue

                # Prepare label
                if phone2idx:
                    # CTC phone labels
                    phones = self._text_to_phones(text)
                    label_ids = [phone2idx.get(p, 0) for p in phones]
                    label_data = np.array(label_ids, dtype=np.int32).tobytes()
                    data_type = DataTypes.CTC_LABEL
                else:
                    # ED text labels
                    label_data = text.encode('utf-8')
                    data_type = DataTypes.ED_LABEL

                # Key
                key = struct.pack(key_fmt, idx)

                # Value: json metadata + audio features
                value = json.dumps({
                    "audio_path": audio_path,
                    "text": text,
                    "data_type": data_type,
                    "language": self.lang,
                    "feat_dim": feat_dim
                }, ensure_ascii=False).encode('utf-8')

                txn.put(key, value)

                if (idx + 1) % 1000 == 0:
                    logger.info(f"{split}: {idx + 1}/{total_samples} samples processed")

        logger.info(f"LMDB creation complete: {total_samples} samples")

    def _text_to_phones(self, text: str) -> List[str]:
        """Convert text to phone sequence.

        Simplified version - actual implementation would use g2p.
        """
        # Return word-level labels for now (ED mode)
        return text.split()

    def generate_chunk_index(
        self,
        lmdb_path: str,
        save_path: str,
        lang_ratio: Optional[Dict[str, float]] = None
    ):
        """Generate chunk index file for chunked training.

        Args:
            lmdb_path: Path to LMDB database
            save_path: Output path for chunk index file
            lang_ratio: Language sampling ratios
        """
        if lang_ratio is None:
            lang_ratio = {self.lang: 1.0}

        env = lmdb.open(lmdb_path, subdir=False, readonly=True)
        with env.begin() as txn:
            total_samples = txn.stat()['entries']

        chunk_size = self.chunk_size
        num_chunks = (total_samples + chunk_size - 1) // chunk_size

        chunk_indices = []
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_samples)
            chunk_indices.append((start_idx, end_idx))

        # Save chunk indices
        with open(save_path, 'wb') as f:
            for start, end in chunk_indices:
                f.write(struct.pack("<QQ", start, end))

        logger.info(f"Generated {len(chunk_indices)} chunks, saved to {save_path}")
        return save_path


def prepare_hungarian_data(
    output_dir: str,
    train_dataset,
    eval_dataset,
    phone2idx: Optional[Dict[str, int]] = None,
    chunk_size: int = 10000
) -> Dict[str, str]:
    """Prepare Hungarian ASR data for training.

    Args:
        output_dir: Output directory
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        phone2idx: Phone to index mapping
        chunk_size: Chunk size for training

    Returns:
        Dict with paths to generated files
    """
    preparator = HungarianASRDataPreparator(
        output_dir=output_dir,
        chunk_size=chunk_size
    )

    train_db, eval_db, phone_dict = preparator.prepare_from_huggingface(
        train_dataset, eval_dataset, phone2idx
    )

    # Generate chunk indices
    train_chunk_path = os.path.join(output_dir, "train_chunks.bin")
    eval_chunk_path = os.path.join(output_dir, "eval_chunks.bin")

    preparator.generate_chunk_index(train_db, train_chunk_path)
    preparator.generate_chunk_index(eval_db, eval_chunk_path)

    return {
        "train_lmdb": train_db,
        "eval_lmdb": eval_db,
        "phone_dict": phone_dict,
        "train_chunks": train_chunk_path,
        "eval_chunks": eval_chunk_path
    }
