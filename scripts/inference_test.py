#!/usr/bin/env python3
"""
Hungarian Whisper Inference Test

Tests the fine-tuned model with Hungarian audio samples.
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
for k in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']:
    os.environ.pop(k, None)

import sys
import logging
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_test_audio(text, duration=3.0, sampling_rate=16000):
    """Generate synthetic audio from text (for testing)."""
    # Simple tone-based audio generation
    np.random.seed(hash(text) % (2**32))
    audio = np.random.randn(int(sampling_rate * duration)).astype(np.float32) * 0.01

    # Add some frequency components
    t = np.linspace(0, duration, int(sampling_rate * duration))
    freq = 200 + (hash(text) % 100)
    audio += 0.1 * np.sin(2 * np.pi * freq * t)

    return audio


def load_model(model_path):
    """Load fine-tuned model."""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    logger.info(f"Loading model from {model_path}")
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    processor = WhisperProcessor.from_pretrained(model_path)

    return model, processor


def transcribe(model, processor, audio, sampling_rate=16000):
    """Transcribe audio."""
    model.eval()

    with torch.no_grad():
        # Process audio
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)

        inputs = processor(
            audio,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        )

        input_features = inputs.input_features.to(model.device)

        # Generate
        generated_ids = model.generate(
            input_features,
            max_new_tokens=448,
            num_beams=1
        )

        # Decode
        transcription = processor.batch_decode(generated_ids)[0]

    return transcription


def run_inference_tests(model_path, test_phrases):
    """Run inference tests with synthetic audio."""
    model, processor = load_model(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    logger.info("\n" + "=" * 60)
    logger.info("Hungarian Whisper Inference Test")
    logger.info("=" * 60)

    results = []
    for i, phrase in enumerate(test_phrases):
        logger.info(f"\n[Test {i+1}] Input text: '{phrase}'")

        # Generate synthetic audio
        audio = generate_test_audio(phrase, duration=3.0)

        # Transcribe
        transcription = transcribe(model, processor, audio)
        logger.info(f"[Result] Transcription: '{transcription}'")

        results.append({
            "input": phrase,
            "transcription": transcription
        })

    return results


def test_with_real_audio(model_path, audio_path):
    """Test with real audio file if available."""
    try:
        import librosa

        model, processor = load_model(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        logger.info(f"\nLoading audio from {audio_path}")
        audio, sr = librosa.load(audio_path, sr=16000)

        transcription = transcribe(model, processor, audio, sr)
        logger.info(f"Transcription: {transcription}")

        return transcription

    except Exception as e:
        logger.error(f"Failed to process audio: {e}")
        return None


def main():
    model_path = "./output/checkpoints_real"

    logger.info("=" * 60)
    logger.info("Hungarian Whisper Inference Test")
    logger.info("=" * 60)

    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        sys.exit(1)

    device = torch.device("cuda")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Test phrases
    test_phrases = [
        "köszönöm szépen",
        "üdvözöllek",
        "hogyan vagy",
        "magyar nyelv",
        "hello world"
    ]

    # Run inference tests
    results = run_inference_tests(model_path, test_phrases)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Inference Test Summary")
    logger.info("=" * 60)
    for i, r in enumerate(results):
        logger.info(f"{i+1}. Input: {r['input']}")
        logger.info(f"   Output: {r['transcription']}")

    logger.info("\nNote: Using synthetic audio - real audio will differ")


if __name__ == "__main__":
    main()
