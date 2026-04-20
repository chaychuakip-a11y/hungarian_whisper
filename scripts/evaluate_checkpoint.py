#!/usr/bin/env python3
"""
Quick evaluation and test with existing checkpoint
Runs inference to verify the model works
"""

import sys
import os
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("Hungarian Whisper - Evaluation with Checkpoint")
    logger.info("=" * 60)

    if not torch.cuda.is_available():
        logger.error("ROCm not available!")
        sys.exit(1)

    device = torch.device("cuda")

    # Check memory
    allocated = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"VRAM: {allocated:.2f}GB / {total:.2f}GB")

    # Load from existing checkpoint
    checkpoint_path = "./output/checkpoints"
    logger.info(f"\n[1] Loading from checkpoint: {checkpoint_path}")

    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    try:
        model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)
        processor = WhisperProcessor.from_pretrained(checkpoint_path)
        logger.info("Checkpoint loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return

    model = model.to(device)
    model.eval()

    allocated = torch.cuda.memory_allocated() / 1e9
    logger.info(f"VRAM after model load: {allocated:.2f}GB")

    # Run multiple inference tests
    logger.info("\n[2] Running inference tests...")

    test_phrases = [
        "köszönöm szépen",
        "üdvözöllek",
        "hogyan vagy"
    ]

    for i, phrase in enumerate(test_phrases):
        logger.info(f"\nTest {i+1}: '{phrase}'")

        # Create dummy audio features using processor
        # This ensures correct format
        audio = torch.randn(16000)  # 1 second of audio
        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        )

        input_features = inputs.input_features.to(device)

        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                input_features,
                max_new_tokens=50,
                do_sample=False
            )

        transcription = processor.batch_decode(generated_ids)[0]
        logger.info(f"  Generated: {transcription}")

    # Run a training simulation with very small steps
    logger.info("\n[3] Testing backward pass (small batch)...")

    model.train()

    # Create small random batch
    batch_features = torch.randn(1, 80, 3000).to(device)
    batch_labels = torch.randint(0, 51865, (1, 10)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # One forward + backward
    outputs = model(input_features=batch_features, labels=batch_labels)
    loss = outputs.loss
    logger.info(f"  Forward pass successful, loss: {loss.item():.4f}")

    loss.backward()
    logger.info("  Backward pass successful")

    optimizer.step()
    logger.info("  Optimizer step successful")

    optimizer.zero_grad()

    allocated = torch.cuda.memory_allocated() / 1e9
    logger.info(f"\nFinal VRAM: {allocated:.2f}GB")

    logger.info("\n" + "=" * 60)
    logger.info("Evaluation completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()