"""Evaluation metrics for Whisper ASR model."""

import logging
from typing import Callable, Dict, Optional

import evaluate
import numpy as np
import torch
from transformers import Seq2SeqTrainer

logger = logging.getLogger(__name__)


def load_wer_metric() -> evaluate.EvaluationModule:
    """Load Word Error Rate metric.

    Returns:
        WER evaluation module.
    """
    return evaluate.load("wer")


def compute_wer_metrics(
    eval_pred,
    tokenizer,
    wer_metric: Optional[evaluate.EvaluationModule] = None
) -> Dict[str, float]:
    """Compute WER from evaluation predictions.

    Args:
        eval_pred: Evaluation predictions from trainer.
        tokenizer: Whisper tokenizer for decoding.
        wer_metric: WER metric instance.

    Returns:
        Dict with WER score.
    """
    if wer_metric is None:
        wer_metric = load_wer_metric()

    predictions, labels = eval_pred

    # Replace -100 in labels (padding)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # WER is computed on lowercase, whitespace-normalized text
    decoded_preds = [pred.lower().strip() for pred in decoded_preds]
    decoded_labels = [label.lower().strip() for label in decoded_labels]

    # Filter empty references
    valid_pairs = [
        (pred, label) for pred, label in zip(decoded_preds, decoded_labels)
        if label
    ]

    if not valid_pairs:
        return {"wer": 0.0}

    filtered_preds, filtered_labels = zip(*valid_pairs)

    wer = wer_metric.compute(
        predictions=list(filtered_preds),
        references=list(filtered_labels)
    )

    return {"wer": wer}


class EvaluationRunner:
    """Runner for model evaluation."""

    def __init__(self, tokenizer, wer_metric: Optional[evaluate.EvaluationModule] = None):
        """Initialize evaluation runner.

        Args:
            tokenizer: Whisper tokenizer.
            wer_metric: Optional pre-loaded WER metric.
        """
        self.tokenizer = tokenizer
        self.wer_metric = wer_metric

    def evaluate(self, trainer: Seq2SeqTrainer, eval_dataset) -> Dict[str, float]:
        """Run evaluation on dataset.

        Args:
            trainer: Seq2SeqTrainer with trained model.
            eval_dataset: Dataset to evaluate on.

        Returns:
            Dict with evaluation metrics.
        """
        logger.info("Running evaluation...")

        predictions = trainer.predict(
            eval_dataset,
            metric_key_prefix="eval",
            max_length=448,
            num_beams=1
        )

        metrics = compute_wer_metrics(
            (predictions.predictions, predictions.label_ids),
            self.tokenizer,
            self.wer_metric
        )

        logger.info(f"Evaluation WER: {metrics['wer']:.4f}")

        return metrics

    def compute_detailed_metrics(
        self,
        predictions: list,
        references: list
    ) -> Dict[str, float]:
        """Compute detailed ASR metrics.

        Args:
            predictions: List of predicted transcriptions.
            references: List of reference transcriptions.

        Returns:
            Dict with WER and additional metrics.
        """
        if self.wer_metric is None:
            self.wer_metric = load_wer_metric()

        # Normalize
        norm_preds = [p.lower().strip() for p in predictions]
        norm_refs = [r.lower().strip() for r in references]

        # WER
        wer = self.wer_metric.compute(
            predictions=norm_preds,
            references=norm_refs
        )

        # Character Error Rate (CER)
        cer_metric = evaluate.load("cer")
        cer = cer_metric.compute(
            predictions=norm_preds,
            references=norm_refs
        )

        return {
            "wer": wer,
            "cer": cer
        }


def create_compute_metrics_func(tokenizer) -> Callable:
    """Create metrics computation function for trainer.

    Args:
        tokenizer: Whisper tokenizer.

    Returns:
        Function compatible with Seq2SeqTrainer compute_metrics.
    """
    wer_metric = load_wer_metric()

    def compute_metrics(eval_pred):
        return compute_wer_metrics(eval_pred, tokenizer, wer_metric)

    return compute_metrics
