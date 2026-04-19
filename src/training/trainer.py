"""Training pipeline for Hungarian Whisper fine-tuning."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.memory_monitor import MemoryMonitor

logger = logging.getLogger(__name__)


def create_training_arguments(
    output_dir: str,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-3,
    num_train_epochs: int = 3,
    fp16: bool = False,
    bf16: bool = True,
    logging_steps: int = 100,
    save_steps: int = 500,
    eval_steps: int = 500,
    warmup_steps: int = 100,
    dataloader_num_workers: int = 4,
    group_by_length: bool = True,
    **kwargs
) -> Seq2SeqTrainingArguments:
    """Create training arguments for Seq2Seq model.

    Args:
        output_dir: Directory for checkpoints and logs.
        per_device_train_batch_size: Batch size per device.
        gradient_accumulation_steps: Gradient accumulation steps.
        learning_rate: Learning rate.
        num_train_epochs: Number of training epochs.
        fp16: Use FP16 precision.
        bf16: Use BF16 precision.
        logging_steps: Log every N steps.
        save_steps: Save checkpoint every N steps.
        eval_steps: Evaluate every N steps.
        warmup_steps: Learning rate warmup steps.
        dataloader_num_workers: DataLoader worker processes.
        group_by_length: Group sequences by length.

    Returns:
        Configured Seq2SeqTrainingArguments.
    """
    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        fp16=fp16,
        bf16=bf16,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        warmup_steps=warmup_steps,
        dataloader_num_workers=dataloader_num_workers,
        group_by_length=group_by_length,
        remove_unused_columns=False,
        label_names=["labels"],
        predict_with_generate=True,
        generation_max_length=448,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=["tensorboard"],
        dataloader_pin_memory=True,
        **kwargs
    )

    return args


def train_whisper(
    model,
    train_dataset,
    eval_dataset,
    feature_extractor,
    tokenizer,
    data_collator,
    output_dir: str = "./output/checkpoints",
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-3,
    num_train_epochs: int = 3,
    warmup_steps: int = 100,
    logging_steps: int = 100,
    save_steps: int = 500,
    eval_steps: int = 500,
    bf16: bool = True,
    fp16: bool = False,
    compute_metrics=None,
    callbacks=None,
    max_vram_gb: float = 24.0
) -> Tuple[Seq2SeqTrainer, Dict]:
    """Train Whisper model with LoRA.

    Args:
        model: PEFT-wrapped Whisper model.
        train_dataset: Training dataset.
        eval_dataset: Evaluation dataset.
        feature_extractor: WhisperFeatureExtractor.
        tokenizer: WhisperTokenizer.
        data_collator: Data collator function.
        output_dir: Output directory for checkpoints.
        per_device_train_batch_size: Batch size per device.
        gradient_accumulation_steps: Gradient accumulation.
        learning_rate: Learning rate.
        num_train_epochs: Number of epochs.
        warmup_steps: Warmup steps.
        logging_steps: Log every N steps.
        save_steps: Save every N steps.
        eval_steps: Evaluate every N steps.
        bf16: Use BF16 precision.
        fp16: Use FP16 precision.
        compute_metrics: Metrics computation function.
        callbacks: Trainer callbacks.
        max_vram_gb: Maximum VRAM in GB for monitoring.

    Returns:
        Tuple of (trainer, train_result)
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create training arguments
    training_args = create_training_arguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        bf16=bf16,
        fp16=fp16
    )

    # Initialize memory monitor
    memory_monitor = MemoryMonitor(max_vram_gb=max_vram_gb)

    # Create trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        callbacks=callbacks
    )

    # Attach memory monitoring
    trainer.add_callback(MemoryMonitorCallback(memory_monitor))

    # Log initial memory state
    memory_monitor.log_memory_stats("Before training")

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()

    # Final memory stats
    memory_monitor.log_memory_stats("After training")

    # Save final model
    trainer.save_model()
    trainer.save_state()

    return trainer, train_result


class MemoryMonitorCallback:
    """Callback to monitor VRAM usage during training."""

    def __init__(self, memory_monitor: MemoryMonitor):
        self.memory_monitor = memory_monitor
        self.step_count = 0

    def on_step_end(self, args, state, control, **kwargs):
        """Log memory usage every N steps."""
        self.step_count += 1
        if self.step_count % 100 == 0:
            self.memory_monitor.log_memory_stats(f"Step {self.step_count}")

            # Check for potential OOM
            if self.memory_monitor.check_oom():
                logger.warning("VRAM usage high - consider reducing batch size")
