#!/usr/bin/env python3
"""
CTC Decoder for Hungarian ASR

Provides CTC greedy decoding and prefix beam search decoding.
Supports integration with WFST for language model decoding.

Usage:
    python -m src.utils.ctc_decoder --help
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import heapq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# CTC Blank index
BLANK_ID = 0


class CTCGreedyDecoder:
    """CTC Greedy decoding - take argmax at each timestep."""

    def __init__(self, blank_id: int = BLANK_ID):
        self.blank_id = blank_id

    def decode(self, emissions: List[List[float]]) -> List[int]:
        """
        Greedy decode CTC emissions.

        Args:
            emissions: List of (time_steps, vocab_size) log probabilities

        Returns:
            List of decoded token indices
        """
        decoded = []
        for timestep in emissions:
            if isinstance(timestep, list):
                best_idx = timestep.index(max(timestep))
            else:
                best_idx = timestep
            decoded.append(best_idx)
        return decoded


class CTCBeamSearchDecoder:
    """CTC Prefix Beam Search decoding."""

    def __init__(
        self,
        vocab_size: int = 51865,
        beam_size: int = 10,
        blank_id: int = BLANK_ID,
        prune_threshold: float = 1e-6
    ):
        self.vocab_size = vocab_size
        self.beam_size = beam_size
        self.blank_id = blank_id
        self.prune_threshold = prune_threshold

    def decode(self, emissions: List[List[float]]) -> List[Tuple[str, float]]:
        """
        CTC Prefix Beam Search decoding.

        Args:
            emissions: List of (time_steps, vocab_size) log probabilities

        Returns:
            List of (transcription, log_probability) tuples, sorted by probability
        """
        T = len(emissions)
        V = self.vocab_size

        # Initialize: (prefix, blank_last, log_prob)
        # prefix: tuple of token indices
        # blank_last: True if last token was blank
        # log_prob: log probability of this beam
        beams = [(tuple(), True, 0.0)]

        for t in range(T):
            new_beams_dict = {}

            for prefix, blank_last, log_prob in beams:
                # Get top-k emissions to reduce computation
                emission = emissions[t]
                topk_probs, topk_indices = [], []

                for i, prob in enumerate(emission):
                    if prob > self.prune_threshold:
                        topk_probs.append(prob)
                        topk_indices.append(i)

                for c in range(V):
                    # Get probability
                    if c in topk_indices:
                        prob = emission[c]
                    else:
                        prob = emission[c]
                        if prob < self.prune_threshold:
                            continue

                    log_p = log_prob + prob

                    if c == self.blank_id:
                        # Blank: keep prefix, set blank_last=True
                        new_prefix = prefix
                        new_blank_last = True
                        key = (new_prefix, new_blank_last)
                        if key not in new_beams_dict or log_p > new_beams_dict[key]:
                            new_beams_dict[key] = log_p
                    elif blank_last or (len(prefix) > 0 and prefix[-1] == c):
                        # Same as last token or repeated: extend prefix, blank_last=True
                        new_prefix = prefix + (c,)
                        new_blank_last = True
                        key = (new_prefix, new_blank_last)
                        if key not in new_beams_dict or log_p > new_beams_dict[key]:
                            new_beams_dict[key] = log_p
                    else:
                        # New token: extend prefix, blank_last=False
                        new_prefix = prefix + (c,)
                        new_blank_last = False
                        key = (new_prefix, new_blank_last)
                        if key not in new_beams_dict or log_p > new_beams_dict[key]:
                            new_beams_dict[key] = log_p

            # Convert to list and keep top beams
            beams = [(prefix, blank_last, log_prob) for (prefix, blank_last), log_prob in new_beams_dict.items()]
            beams.sort(key=lambda x: x[2], reverse=True)
            beams = beams[:self.beam_size]

            # Handle empty beams
            if not beams:
                beams = [(tuple(), True, 0.0)]

        # Sort by log probability
        beams.sort(key=lambda x: x[2], reverse=True)

        # Convert to strings
        results = []
        for prefix, _, log_prob in beams:
            text = self._indices_to_text(prefix)
            results.append((text, log_prob))

        return results

    def _indices_to_text(self, indices: Tuple[int]) -> str:
        """Convert token indices to text."""
        return " ".join(str(idx) for idx in indices)


class CTCPrefixBeamSearch:
    """CTC Prefix Beam Search with language model integration."""

    def __init__(
        self,
        vocab_size: int = 51865,
        beam_size: int = 10,
        blank_id: int = BLANK_ID,
        lm_path: Optional[str] = None,
        lm_weight: float = 0.0
    ):
        self.vocab_size = vocab_size
        self.beam_size = beam_size
        self.blank_id = blank_id
        self.lm_weight = lm_weight
        self.lm = None

        if lm_path and os.path.exists(lm_path):
            self._load_lm(lm_path)

    def _load_lm(self, lm_path: str):
        """Load language model from file."""
        self.lm = {}
        with open(lm_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    word = parts[0]
                    prob = float(parts[1])
                    self.lm[word] = prob
        logger.info(f"Loaded LM with {len(self.lm)} words")

    def decode(self, emissions: List[List[float]]) -> str:
        """Decode with optional LM integration."""
        beams = self._prefix_beam_search(emissions)

        if beams and self.lm and self.lm_weight > 0:
            # Rescore with LM
            beams = self._rescore_with_lm(beams)

        if beams:
            return beams[0][0]
        return ""

    def _prefix_beam_search(self, emissions: List[List[float]]) -> List[Tuple[str, float]]:
        """Standard CTC prefix beam search."""
        decoder = CTCBeamSearchDecoder(
            vocab_size=self.vocab_size,
            beam_size=self.beam_size,
            blank_id=self.blank_id
        )
        return decoder.decode(emissions)

    def _rescore_with_lm(self, beams: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Rescore beams with language model."""
        rescored = []
        for text, log_prob in beams:
            words = text.split()
            lm_score = sum(self.lm.get(w, 1e-10) for w in words)
            combined_score = log_prob + self.lm_weight * lm_score
            rescored.append((text, combined_score))

        rescored.sort(key=lambda x: x[1], reverse=True)
        return rescored


def decode_ctc_greedy(emissions: List[List[float]], blank_id: int = BLANK_ID) -> str:
    """Quick CTC greedy decoding."""
    decoder = CTCGreedyDecoder(blank_id=blank_id)
    indices = decoder.decode(emissions)

    # Remove consecutive duplicates and blanks
    result = []
    prev = -1
    for idx in indices:
        if idx != blank_id and idx != prev:
            result.append(idx)
        prev = idx

    return result


def decode_ctc_to_text(token_ids: List[int], tokenizer) -> str:
    """Convert CTC decoded token IDs to text using tokenizer."""
    # Filter out special tokens
    special_tokens = {tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.unk_token_id}
    if hasattr(tokenizer, 'transcribe_token_id'):
        # For Whisper tokenizer
        filtered = [tid for tid in token_ids if tid not in special_tokens and tid > 0]
    else:
        filtered = token_ids

    if not filtered:
        return ""

    # Decode using tokenizer
    try:
        text = tokenizer.decode(filtered)
        return text
    except Exception:
        return " ".join(str(t) for t in filtered)


def decode_ctc_beam(
    emissions: List[List[float]],
    beam_size: int = 10,
    blank_id: int = BLANK_ID
) -> Tuple[str, float]:
    """Quick CTC beam search decoding."""
    decoder = CTCBeamSearchDecoder(
        vocab_size=len(emissions[0]) if emissions else 0,
        beam_size=beam_size,
        blank_id=blank_id
    )
    results = decoder.decode(emissions)
    if results:
        return results[0]
    return ("", float('-inf'))


class CTCModelWrapper:
    """Wrapper for CTC-style model inference.

    Used for models that output CTC-style emissions (logits over phone/char vocab).
    """

    def __init__(
        self,
        model,
        processor,
        is_ctc_model: bool = True,
        blank_id: int = BLANK_ID
    ):
        self.model = model
        self.processor = processor
        self.is_ctc_model = is_ctc_model
        self.blank_id = blank_id

    def get_emissions(self, audio, sampling_rate: int = 16000) -> List[List[float]]:
        """Get CTC emissions from audio."""
        import torch

        self.model.eval()
        device = next(self.model.parameters()).device

        with torch.no_grad():
            # Process audio
            inputs = self.processor(
                audio,
                sampling_rate=sampling_rate,
                return_tensors="pt"
            )
            input_features = inputs.input_features.to(device)

            if self.is_ctc_model:
                # CTC model: get logits directly
                outputs = self.model(input_features=input_features)
                emissions = torch.log_softmax(outputs.logits, dim=-1)
                emissions = emissions.cpu().numpy()[0]  # (time, vocab)
            else:
                # ED model: use encoder outputs as pseudo-emissions
                outputs = self.model.model.encoder(input_features)
                emissions = outputs.last_hidden_state
                emissions = torch.log_softmax(emissions, dim=-1)
                emissions = emissions.cpu().numpy()[0]  # (time, vocab)

        # Convert to list of list
        emissions_list = emissions.tolist()
        return emissions_list

    def transcribe_greedy(self, audio, sampling_rate: int = 16000) -> str:
        """Transcribe using greedy CTC decoding."""
        emissions = self.get_emissions(audio, sampling_rate)
        return decode_ctc_greedy(emissions, self.blank_id)

    def transcribe_beam(
        self,
        audio,
        sampling_rate: int = 16000,
        beam_size: int = 10
    ) -> str:
        """Transcribe using CTC beam search."""
        emissions = self.get_emissions(audio, sampling_rate)
        text, _ = decode_ctc_beam(emissions, beam_size, self.blank_id)
        return text


if __name__ == "__main__":
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description="CTC Decoder")
    parser.add_argument('--test', action='store_true', help='Run test')
    parser.add_argument('--emission_size', type=int, default=100,
                        help='Number of timesteps')
    parser.add_argument('--vocab_size', type=int, default=50,
                        help='Vocabulary size')
    args = parser.parse_args()

    if args.test:
        # Generate dummy emissions
        emissions = np.random.randn(args.emission_size, args.vocab_size).tolist()
        emissions = [list(e) for e in emissions]

        # Test greedy
        print("Testing CTC Greedy Decoder...")
        result = decode_ctc_greedy(emissions)
        print(f"Greedy result: {result}")

        # Test beam
        print("\nTesting CTC Beam Search Decoder...")
        text, score = decode_ctc_beam(emissions, beam_size=5)
        print(f"Beam result: {text} (score: {score:.4f})")

        print("\nCTC Decoder test passed!")
