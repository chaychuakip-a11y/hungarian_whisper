"""LoRA configuration for Whisper model fine-tuning."""

import logging
from typing import Optional

import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperTokenizer
)
from peft import LoraConfig, get_peft_model, TaskType

logger = logging.getLogger(__name__)


class LoRAWhisperModel:
    """Whisper model with LoRA fine-tuning configuration.

    Args:
        model_name: HuggingFace model name (e.g., "openai/whisper-large-v2").
        lora_r: LoRA attention dimension (rank).
        lora_alpha: LoRA alpha parameter.
        lora_dropout: LoRA dropout probability.
        target_modules: List of module names to apply LoRA to.
        int8: Whether to load model in 8-bit precision.
        device_map: Device mapping strategy.
    """

    def __init__(
        self,
        model_name: str = "openai/whisper-large-v2",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[list] = None,
        int8: bool = True,
        device_map: str = "auto"
    ):
        self.model_name = model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        self.int8 = int8
        self.device_map = device_map

        self.model = None
        self.processor = None
        self.feature_extractor = None
        self.tokenizer = None

    def load_model(self) -> WhisperForConditionalGeneration:
        """Load Whisper model with LoRA and optional INT8 quantization.

        Returns:
            PEFT-wrapped Whisper model ready for training.
        """
        logger.info(f"Loading model: {self.model_name}")

        # Load base model
        load_kwargs = {
            "pretrained_model_name_or_path": self.model_name,
            "device_map": self.device_map
        }

        if self.int8:
            load_kwargs["load_in_8bit"] = True

        model = WhisperForConditionalGeneration.from_pretrained(**load_kwargs)

        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias="none"
        )

        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.train()

        self.model = model
        self._print_trainable_parameters()

        return model

    def load_processor(self) -> WhisperProcessor:
        """Load Whisper processor (feature extractor + tokenizer).

        Returns:
            WhisperProcessor for feature extraction and tokenization.
        """
        logger.info(f"Loading processor: {self.model_name}")

        processor = WhisperProcessor.from_pretrained(self.model_name)
        self.processor = processor
        self.feature_extractor = processor.feature_extractor
        self.tokenizer = processor.tokenizer

        return processor

    def _print_trainable_parameters(self) -> None:
        """Log the number of trainable parameters."""
        trainable_params = 0
        all_params = 0

        for _, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        trainable_percent = 100 * trainable_params / all_params

        logger.info(
            f"Trainable params: {trainable_params:,} / {all_params:,} "
            f"({trainable_percent:.2f}%)"
        )

    def get_model(self) -> WhisperForConditionalGeneration:
        """Get the loaded model."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model

    def get_feature_extractor(self) -> WhisperFeatureExtractor:
        """Get the feature extractor."""
        if self.feature_extractor is None:
            self.load_processor()
        return self.feature_extractor

    def get_tokenizer(self) -> WhisperTokenizer:
        """Get the tokenizer."""
        if self.tokenizer is None:
            self.load_processor()
        return self.tokenizer


def create_lora_whisper(
    model_name: str = "openai/whisper-large-v2",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[list] = None,
    int8: bool = True
) -> tuple:
    """Create a LoRA-wrapped Whisper model with processor.

    Args:
        model_name: Model name on HuggingFace.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha.
        lora_dropout: Dropout probability.
        target_modules: Modules for LoRA attachment.
        int8: Use INT8 quantization.

    Returns:
        Tuple of (model, feature_extractor, tokenizer)
    """
    lora_whisper = LoRAWhisperModel(
        model_name=model_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        int8=int8
    )

    model = lora_whisper.load_model()
    processor = lora_whisper.load_processor()

    return model, processor.feature_extractor, processor.tokenizer
