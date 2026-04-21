#!/usr/bin/env python3
"""
Hungarian Whisper Real Audio Inference

Usage:
    # With synthetic audio (for testing):
    python scripts/inference_real.py --mode synthetic

    # With real audio file:
    python scripts/inference_real.py --mode file --audio_path /path/to/audio.wav

    # With streaming dataset (downloads one sample):
    python scripts/inference_real.py --mode dataset
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
for k in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']:
    os.environ.pop(k, None)

import sys
import logging
from pathlib import Path
import argparse

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    device = next(model.parameters()).device

    with torch.no_grad():
        # Convert to numpy if needed
        if isinstance(audio, np.ndarray):
            pass  # Already numpy
        elif hasattr(audio, 'cpu'):
            audio = audio.cpu().numpy()

        inputs = processor(
            audio,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        )

        input_features = inputs.input_features.to(device)

        generated_ids = model.generate(
            input_features,
            max_new_tokens=448,
            num_beams=1
        )

        transcription = processor.batch_decode(generated_ids)[0]

    return transcription


def test_synthetic(model, processor):
    """Test with synthetic audio."""
    test_phrases = [
        "köszönöm szépen",
        "üdvözöllek",
        "hogyan vagy",
        "magyar nyelv"
    ]

    logger.info("\n" + "=" * 60)
    logger.info("Synthetic Audio Test")
    logger.info("=" * 60)

    for i, phrase in enumerate(test_phrases):
        # Generate synthetic audio
        np.random.seed(hash(phrase) % (2**32))
        duration = 3.0
        audio = np.random.randn(int(16000 * duration)).astype(np.float32) * 0.01

        # Add some frequency
        t = np.linspace(0, duration, int(16000 * duration))
        freq = 200 + (hash(phrase) % 100)
        audio += 0.1 * np.sin(2 * np.pi * freq * t)

        transcription = transcribe(model, processor, audio)
        logger.info(f"[{i+1}] '{phrase}' -> '{transcription}'")

    return True


def test_file(model, processor, audio_path):
    """Test with audio file."""
    import librosa

    logger.info(f"\nLoading audio from {audio_path}")
    audio, sr = librosa.load(audio_path, sr=16000)

    transcription = transcribe(model, processor, audio, sr)
    logger.info(f"Transcription: {transcription}")

    return transcription


def test_dataset(model, processor, dataset_name="facebook/voxpopuli", subset="hu"):
    """Test with one sample from streaming dataset."""
    from datasets import load_dataset

    logger.info(f"\nDownloading one sample from {dataset_name}/{subset}...")

    try:
        ds = load_dataset(dataset_name, subset, split='train', streaming=True)
        sample = next(iter(ds))

        audio = sample['audio']['array']
        sr = sample['audio']['sampling_rate']
        text = sample.get('text', sample.get('sentence', 'N/A'))

        logger.info(f"Sample text: '{text}'")
        logger.info(f"Audio: {audio.shape}, SR: {sr}")

        # Resample if needed
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        transcription = transcribe(model, processor, audio, 16000)
        logger.info(f"Transcription: {transcription}")

        return transcription

    except Exception as e:
        logger.error(f"Failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Hungarian Whisper Inference')
    parser.add_argument('--model_path', type=str, default='./output/checkpoints_real',
                        help='Path to trained model')
    parser.add_argument('--mode', type=str, default='synthetic',
                        choices=['synthetic', 'file', 'dataset'],
                        help='Test mode')
    parser.add_argument('--audio_path', type=str, default=None,
                        help='Audio file path (for file mode)')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Hungarian Whisper Real Audio Inference")
    logger.info("=" * 60)

    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        sys.exit(1)

    device = torch.device("cuda")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    model, processor = load_model(args.model_path)
    model = model.to(device)

    # Run test
    if args.mode == 'synthetic':
        test_synthetic(model, processor)
    elif args.mode == 'file':
        if not args.audio_path:
            logger.error("No audio_path specified!")
            sys.exit(1)
        test_file(model, processor, args.audio_path)
    elif args.mode == 'dataset':
        test_dataset(model, processor)

    logger.info("\n" + "=" * 60)
    logger.info("Inference Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
