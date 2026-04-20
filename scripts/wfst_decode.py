#!/usr/bin/env python3
"""
CTC/ED WFST Decoding Script

Provides WFST-based beam search decoding for Hungarian ASR models.
Supports HCLG graph construction and prefix beam search.

Usage:
    # Build WFST graph
    python scripts/wfst_decode.py --build_graph --output_dir ./data/wfst

    # Decode with trained model
    python scripts/wfst_decode.py \
        --model_path ./output/checkpoints \
        --audio_path /path/to/audio.wav \
        --output_dir ./data/wfst
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import argparse
import json
import logging

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.wfst_decoder import (
    WFSTDecoder,
    CTCPrefixDecoder,
    build_decoding_graph,
    HUNGARIAN_PHONES,
    PHONE2IDX,
    IDX2PHONE
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(model_path: str):
    """Load fine-tuned Whisper model."""
    try:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        logger.info(f"Loading model from {model_path}")
        model = WhisperForConditionalGeneration.from_pretrained(model_path)
        processor = WhisperProcessor.from_pretrained(model_path)

        return model, processor
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None, None


def load_audio(audio_path: str, sampling_rate: int = 16000):
    """Load and preprocess audio."""
    import librosa
    import numpy as np

    logger.info(f"Loading audio from {audio_path}")
    audio, sr = librosa.load(audio_path, sr=sampling_rate)

    return audio


def extract_features(audio, processor):
    """Extract mel spectrogram features."""
    input_features = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features

    return input_features


def ctc_decode_with_wfst(
    emissions: List[List[float]],
    decoder: WFSTDecoder,
    use_prefix_beam: bool = True
) -> str:
    """Decode CTC emissions using WFST.

    Args:
        emissions: Frame-level probability distributions
        decoder: WFST decoder instance
        use_prefix_beam: Use prefix beam search instead of greedy

    Returns:
        Decoded transcription
    """
    if use_prefix_beam:
        ctc_decoder = CTCPrefixDecoder(beam_size=10)
        beams = ctc_decoder.prefix_beam_search(emissions)
        if beams:
            return beams[0][0]
        return ""
    else:
        phones = decoder.decode(emissions, phone_labels=True)
        return " ".join(phones)


def transcribe_audio(
    model,
    processor,
    audio_path: str,
    decoder: Optional[WFSTDecoder] = None,
    use_wfst: bool = False
) -> Dict:
    """Transcribe audio file.

    Args:
        model: Whisper model
        processor: Whisper processor
        audio_path: Path to audio file
        decoder: Optional WFST decoder
        use_wfst: Use WFST decoding instead of default

    Returns:
        Dictionary with transcription and metadata
    """
    # Load audio
    audio = load_audio(audio_path)

    # Extract features
    input_features = extract_features(audio, processor)

    # Generate transcription
    if use_wfst and decoder:
        # CTC-style decoding with WFST
        import torch
        model.eval()
        with torch.no_grad():
            # Get encoder outputs
            encoder_outputs = model.model.encoder(input_features.to(model.device))
            emissions = encoder_outputs.last_hidden_state.mean(dim=1).tolist()

        transcription = ctc_decode_with_wfst(emissions, decoder)
    else:
        # Use model's native decoding
        import torch
        model.eval()
        with torch.no_grad():
            forced_decoder = "ctc" if hasattr(model, 'generate') else None
            # Standard generation
            generated_ids = model.generate(
                input_features.to(model.device),
                max_new_tokens=448
            )
            transcription = processor.batch_decode(generated_ids)[0]

    return {
        "audio_path": audio_path,
        "transcription": transcription,
        "language": "hungarian"
    }


def batch_transcribe(
    model,
    processor,
    audio_dir: str,
    output_path: str,
    decoder: Optional[WFSTDecoder] = None,
    use_wfst: bool = False
) -> None:
    """Transcribe all audio files in a directory.

    Args:
        model: Whisper model
        processor: Whisper processor
        audio_dir: Directory containing audio files
        output_path: Path to save transcriptions JSON
        decoder: Optional WFST decoder
        use_wfst: Use WFST decoding
    """
    audio_dir = Path(audio_dir)
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}

    audio_files = [
        f for f in audio_dir.iterdir()
        if f.suffix.lower() in audio_extensions
    ]

    results = []
    for i, audio_path in enumerate(audio_files):
        logger.info(f"Processing {i+1}/{len(audio_files)}: {audio_path.name}")
        result = transcribe_audio(model, processor, str(audio_path), decoder, use_wfst)
        results.append(result)

    # Save results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Transcriptions saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="WFST Decoding for CTC/ED models")
    parser.add_argument('--model_path', type=str, default='./output/checkpoints',
                        help='Path to trained model')
    parser.add_argument('--audio_path', type=str, default=None,
                        help='Path to audio file')
    parser.add_argument('--audio_dir', type=str, default=None,
                        help='Directory with audio files')
    parser.add_argument('--output_dir', type=str, default='./data/wfst',
                        help='Output directory for WFST graphs')
    parser.add_argument('--output_json', type=str, default='./output/transcriptions.json',
                        help='Output JSON path for transcriptions')
    parser.add_argument('--build_graph', action='store_true',
                        help='Build WFST decoding graph')
    parser.add_argument('--use_wfst', action='store_true',
                        help='Use WFST decoding instead of model default')
    parser.add_argument('--lexicon', type=str, default=None,
                        help='Path to lexicon file')
    parser.add_argument('--lm', type=str, default=None,
                        help='Path to language model file')

    args = parser.parse_args()

    # Build WFST graph if requested
    if args.build_graph:
        paths = build_decoding_graph(args.output_dir, args.lexicon, args.lm)
        print("Generated WFST files:")
        for name, path in paths.items():
            print(f"  {name}: {path}")
        return

    # Load model
    model, processor = load_model(args.model_path)
    if model is None:
        logger.error("Failed to load model")
        sys.exit(1)

    # Initialize WFST decoder if needed
    decoder = None
    if args.use_wfst:
        decoder = WFSTDecoder(args.output_dir)
        if not decoder.load_graph():
            logger.warning("WFST graph not available, using model default")
            decoder = None

    # Transcribe
    if args.audio_path:
        result = transcribe_audio(model, processor, args.audio_path, decoder, args.use_wfst)
        print(f"Transcription: {result['transcription']}")

    elif args.audio_dir:
        batch_transcribe(model, processor, args.audio_dir, args.output_json, decoder, args.use_wfst)

    else:
        print("No audio input specified. Use --audio_path or --audio_dir")


if __name__ == "__main__":
    main()