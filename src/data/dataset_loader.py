"""Dataset loader for Hungarian speech datasets from HuggingFace.

Loads Common Voice, VoxPopuli, and FLEURS datasets and applies
Hungarian-specific filtering and normalization.
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import datasets
from datasets import Audio, concatenate_datasets

from .hungarian_normalizer import HungarianTextNormalizer

logger = logging.getLogger(__name__)


# Dataset configurations
DATASET_CONFIGS = {
    "common_voice": {
        "name": "mozilla-foundation/common_voice",
        "config_name": "hu",
        "split": "train",
        "audio_column": "audio",
        "text_column": "sentence",
    },
    "voxpopuli": {
        "name": "facebook/voxpopuli",
        "config_name": "hu",
        "split": "train",
        "audio_column": "audio",
        "text_column": "normalized_text",
    },
    "fleurs": {
        "name": "google/fleurs",
        "config_name": "hu_hu",
        "split": "train",
        "audio_column": "audio",
        "text_column": "transcription",
    }
}


class HungarianDatasetLoader:
    """Loads and preprocesses Hungarian speech datasets.

    Args:
        cache_dir: Directory for dataset cache.
        min_duration: Minimum audio duration in seconds.
        max_duration: Maximum audio duration in seconds.
    """

    def __init__(
        self,
        cache_dir: str = "./data",
        min_duration: float = 1.0,
        max_duration: float = 25.0
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.normalizer = HungarianTextNormalizer()

    def load_dataset(
        self,
        dataset_name: str,
        trust_remote_code: bool = True
    ) -> datasets.DatasetDict:
        """Load a single Hungarian dataset.

        Args:
            dataset_name: One of 'common_voice', 'voxpopuli', 'fleurs'.
            trust_remote_code: Whether to trust remote code.

        Returns:
            HuggingFace DatasetDict with train/validation splits.
        """
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        config = DATASET_CONFIGS[dataset_name]
        logger.info(f"Loading {dataset_name} dataset...")

        # Load dataset
        raw_dataset = datasets.load_dataset(
            config["name"],
            config["config_name"],
            split=config["split"],
            trust_remote_code=trust_remote_code,
            cache_dir=str(self.cache_dir / dataset_name)
        )

        # Rename columns
        if config["audio_column"] != "audio":
            raw_dataset = raw_dataset.rename_column(config["audio_column"], "audio")
        if config["text_column"] != "text":
            raw_dataset = raw_dataset.rename_column(config["text_column"], "text")

        # Cast audio column
        raw_dataset = raw_dataset.cast_column("audio", Audio(sampling_rate=16000))

        # Filter and normalize
        filtered = self._filter_and_normalize(raw_dataset, dataset_name)

        logger.info(f"Loaded {len(filtered)} samples from {dataset_name}")
        return filtered

    def load_all_datasets(
        self,
        train_ratio: float = 0.9
    ) -> Tuple[datasets.Dataset, datasets.Dataset]:
        """Load all Hungarian datasets and combine them.

        Args:
            train_ratio: Ratio of data to use for training.

        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        all_filtered = []

        for dataset_name in DATASET_CONFIGS.keys():
            try:
                dataset = self.load_dataset(dataset_name)
                all_filtered.append(dataset)
            except Exception as e:
                logger.warning(f"Failed to load {dataset_name}: {e}")
                continue

        if not all_filtered:
            raise RuntimeError("No datasets could be loaded")

        # Concatenate all datasets
        combined = concatenate_datasets(all_filtered)
        logger.info(f"Combined dataset size: {len(combined)}")

        # Shuffle and split
        combined = combined.shuffle(seed=42)

        split_idx = int(len(combined) * train_ratio)
        train_dataset = combined.select(range(split_idx))
        eval_dataset = combined.select(range(split_idx, len(combined)))

        logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

        return train_dataset, eval_dataset

    def _filter_and_normalize(
        self,
        dataset: datasets.Dataset,
        dataset_name: str
    ) -> datasets.Dataset:
        """Filter by duration and normalize text.

        Args:
            dataset: Raw dataset.
            dataset_name: Name for logging.

        Returns:
            Filtered and normalized dataset.
        """
        before_count = len(dataset)

        # Filter by duration
        dataset = dataset.filter(
            self._check_duration,
            input_columns=["audio"],
            desc=f"Filtering {dataset_name} by duration"
        )

        # Normalize text
        dataset = dataset.map(
            self._normalize_text,
            input_columns=["text"],
            desc=f"Normalizing {dataset_name} text",
            load_from_cache_file=False
        )

        # Filter by normalized text validity
        dataset = dataset.filter(
            lambda x: self.normalizer.is_valid_transcription(x),
            input_columns=["normalized_text"],
            desc=f"Filtering {dataset_name} by text validity"
        )

        after_count = len(dataset)
        logger.info(f"{dataset_name}: {before_count} -> {after_count} (removed {before_count - after_count})")

        return dataset

    def _check_duration(self, audio_dict: Dict) -> bool:
        """Check if audio duration is within bounds."""
        duration = audio_dict.get("duration", 0)
        return self.min_duration <= duration <= self.max_duration

    def _normalize_text(self, text: str) -> Dict:
        """Normalize text and return with key."""
        normalized = self.normalizer.normalize(text)
        return {"normalized_text": normalized}


def prepare_hungarian_dataset(
    output_dir: str,
    cache_dir: str = "./data",
    min_duration: float = 1.0,
    max_duration: float = 25.0,
    train_ratio: float = 0.9
) -> Tuple[datasets.Dataset, datasets.Dataset]:
    """Prepare Hungarian dataset for training.

    Args:
        output_dir: Directory to save processed datasets.
        cache_dir: HuggingFace cache directory.
        min_duration: Minimum audio duration.
        max_duration: Maximum audio duration.
        train_ratio: Train/eval split ratio.

    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    loader = HungarianDatasetLoader(
        cache_dir=cache_dir,
        min_duration=min_duration,
        max_duration=max_duration
    )

    train_dataset, eval_dataset = loader.load_all_datasets(train_ratio=train_ratio)

    # Save processed datasets
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_dataset.save_to_disk(str(output_path / "train"))
    eval_dataset.save_to_disk(str(output_path / "eval"))

    logger.info(f"Saved datasets to {output_path}")

    return train_dataset, eval_dataset
