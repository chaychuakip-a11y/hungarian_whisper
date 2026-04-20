"""Data collator for Whisper speech-to-text training."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import WhisperFeatureExtractor, WhisperTokenizer


@dataclass
class HTKDataCollator:
    """Collator for HTK dataset batches.

    Pads input_features and labels to max length in batch.
    """

    tokenizer: WhisperTokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of samples.

        Args:
            features: List of dicts from HTKHungarianDataset.

        Returns:
            Dict with batched input_features and labels tensors.
        """
        # Stack input_features
        input_features = [f["input_features"] for f in features]
        batched_input = torch.stack(input_features)

        # Pad labels
        labels = [f["labels"] for f in features]
        batched_labels = self.tokenizer.pad(
            {"input_ids": labels},
            return_tensors="pt"
        )["input_ids"]

        # Replace padding token ids with -100 for loss computation
        batched_labels = batched_labels.masked_fill(
            batched_labels == self.tokenizer.pad_token_id,
            -100
        )

        return {
            "input_features": batched_input,
            "labels": batched_labels
        }


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator for speech sequence-to-sequence models.

    Used with HuggingFace datasets.
    """

    feature_extractor: WhisperFeatureExtractor
    tokenizer: WhisperTokenizer
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of speech samples.

        Args:
            features: List of dicts with 'input_features' and 'labels'.

        Returns:
            Dict with batched and padded tensors.
        """
        # Extract input features
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        batched_input = self.feature_extractor.pad(
            input_features,
            return_tensors="pt"
        )

        # Extract labels
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.tokenizer.pad(
            label_features,
            return_tensors="pt",
            padding=self.padding,
            max_length=self.max_length_labels,
            pad_to_multiple_of=self.pad_to_multiple_of_labels
        )

        # Replace padding tokens
        batched_labels = labels_batch["input_ids"]
        batched_labels = batched_labels.masked_fill(
            batched_labels == self.tokenizer.pad_token_id,
            -100
        )

        return {
            "input_features": batched_input["input_features"],
            "labels": batched_labels
        }
