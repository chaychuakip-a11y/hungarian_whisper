#!/usr/bin/env python3
"""
CTC Decoding Script for Hungarian ASR

Tests CTC decoding with the fine-tuned Whisper model.

Usage:
    python scripts/ctc_inference.py --model_path ./output/checkpoints_fleurs \
        --audio_path ./data/fleurs_hungarian_sample.wav
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

from utils.ctc_decoder import (
    CTCGreedyDecoder,
    CTCBeamSearchDecoder,
    CTCPrefixBeamSearch,
    decode_ctc_greedy,
    decode_ctc_to_text,
    decode_ctc_beam,
    CTCModelWrapper,
    BLANK_ID
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(model_path):
    """Load fine-tuned model."""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    logger.info(f"Loading model from {model_path}")
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    processor = WhisperProcessor.from_pretrained(model_path)

    return model, processor


def transcribe_with_ctc(
    model,
    processor,
    audio_path: str,
    decode_mode: str = "greedy"
):
    """Transcribe using CTC decoding.

    Args:
        model: Whisper model
        processor: Whisper processor
        audio_path: Path to audio file
        decode_mode: "greedy" or "beam"
    """
    import librosa

    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)

    # Since this is an ED model, we use encoder outputs as pseudo-emissions
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        )
        input_features = inputs.input_features.to(device)

        # Get encoder outputs (pseudo-emissions for CTC decoding)
        encoder_outputs = model.model.encoder(input_features)
        emissions = encoder_outputs.last_hidden_state
        emissions = torch.log_softmax(emissions, dim=-1)
        emissions = emissions.cpu().numpy()[0]  # (time, vocab)

    emissions_list = emissions.tolist()

    # CTC decode
    if decode_mode == "greedy":
        token_ids = decode_ctc_greedy(emissions_list)
        result = decode_ctc_to_text(token_ids, processor.tokenizer)
        logger.info(f"Greedy CTC result: {result}")
    elif decode_mode == "beam":
        result, score = decode_ctc_beam(emissions_list, beam_size=10)
        result = decode_ctc_to_text([int(t) for t in result.split()], processor.tokenizer)
        logger.info(f"Beam CTC result: {result} (score: {score:.4f})")
    else:
        token_ids = decode_ctc_greedy(emissions_list)
        result = decode_ctc_to_text(token_ids, processor.tokenizer)
        logger.info(f"Default CTC result: {result}")

    return result


def main():
    parser = argparse.ArgumentParser(description='CTC Decoding for Hungarian ASR')
    parser.add_argument('--model_path', type=str, default='./output/checkpoints_fleurs',
                        help='Path to trained model')
    parser.add_argument('--audio_path', type=str, default='./data/fleurs_hungarian_sample.wav',
                        help='Path to audio file')
    parser.add_argument('--decode_mode', type=str, default='greedy',
                        choices=['greedy', 'beam'],
                        help='CTC decoding mode')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("CTC Decoding Test")
    logger.info("=" * 60)

    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        sys.exit(1)

    device = torch.device("cuda")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    model, processor = load_model(args.model_path)
    model = model.to(device)

    # Load ground truth
    gt_path = args.audio_path.replace('.wav', '.txt')
    if os.path.exists(gt_path):
        with open(gt_path, 'r') as f:
            ground_truth = f.read().strip()
        logger.info(f"Ground truth: {ground_truth}")

    # CTC decoding
    logger.info(f"\nRunning CTC {args.decode_mode} decoding...")
    result = transcribe_with_ctc(model, processor, args.audio_path, args.decode_mode)

    logger.info("\n" + "=" * 60)
    logger.info("Result")
    logger.info("=" * 60)
    logger.info(f"Decoded: {result}")
    if 'ground_truth' in locals():
        logger.info(f"Ground truth: {ground_truth}")

    logger.info("\n" + "=" * 60)
    logger.info("CTC Decoding Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
