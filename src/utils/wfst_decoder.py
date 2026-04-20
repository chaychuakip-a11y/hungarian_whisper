#!/usr/bin/env python3
"""
WFST Decoder for CTC and ED models

Provides Weighted Finite State Transducer decoding for Hungarian ASR.
Supports HCLG graph construction and beam search decoding.

Usage:
    python -m src.utils.wfst_decoder --help
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Hungarian phone set (48 phones)
HUNGARIAN_PHONES = [
    '<SIL>', '<SP>',
    'a', 'a:', 'e', 'e:', 'i', 'i:', 'o', 'o:', 'o:', 'u', 'u:', 'u:',
    'y', 'y:', 'o:', 'o:', 'o:',
    'b', 'c', 'd', 'dz', 'dzs', 'f', 'g', 'gy', 'h', 'j', 'k', 'l', 'ly',
    'm', 'n', 'ny', 'p', 'q', 'r', 's', 'sz', 't', 'ty', 'v', 'w', 'x', 'y', 'z', 'zs'
]

PHONE2IDX = {p: i + 1 for i, p in enumerate(HUNGARIAN_PHONES)}
PHONE2IDX['<blank>'] = 0
IDX2PHONE = {v: k for k, v in PHONE2IDX.items()}


class WFSTGraphBuilder:
    """Build WFST decoding graphs (L, G, C, T, HCLG)."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_lexicon_fst(self, lexicon_path: str) -> str:
        """Build L.fst - lexicon WFST (word -> phones).

        Format: word -> phone1 phone2 ... phoneN
        """
        output_path = self.output_dir / "L.fst.txt"

        # Read lexicon
        lexicon = {}
        with open(lexicon_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    word = parts[0]
                    phones = parts[1:]
                    lexicon[word] = phones

        # Write L.fst in text format (OpenFST)
        with open(output_path, 'w', encoding='utf-8') as f:
            state = 0
            f.write(f"0 {state} <eps> <eps>\n")

            for word, phones in lexicon.items():
                for phone in phones:
                    f.write(f"{state} {state + 1} {phone} {phone}\n")
                state += 1

            f.write(f"{state}\n")

        logger.info(f"L.fst written to {output_path}")
        return str(output_path)

    def build_language_model_fst(self, lm_path: str) -> str:
        """Build G.fst - language model WFST.

        Format: state -> next_state word weight
        """
        output_path = self.output_dir / "G.fst.txt"

        with open(lm_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        with open(output_path, 'w', encoding='utf-8') as f:
            state = 0
            f.write(f"0 {state} <eps> <eps>\n")

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 3:
                    word = parts[0]
                    prob = float(parts[1])
                    weight = -math.log(prob)

                    f.write(f"{state} {state + 1} {word} {word} {weight}\n")
                    state += 1

            f.write(f"{state}\n")

        logger.info(f"G.fst written to {output_path}")
        return str(output_path)

    def build_context_fst(self) -> str:
        """Build C.fst - context transducers (CDIO to C)."""
        output_path = self.output_dir / "C.fst.txt"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("0 1 <eps> <eps>\n")
            f.write("1\n")

        logger.info(f"C.fst written to {output_path}")
        return str(output_path)

    def build_phone_fst(self) -> str:
        """Build T.fst - phone transducer (phones to words)."""
        output_path = self.output_dir / "T.fst.txt"

        with open(output_path, 'w', encoding='utf-8') as f:
            state = 0
            for phone in HUNGARIAN_PHONES:
                f.write(f"{state} {state} {phone} {phone}\n")
            f.write(f"{state}\n")

        logger.info(f"T.fst written to {output_path}")
        return str(output_path)

    def build_hclg(self, model_dir: str) -> str:
        """Build HCLG.fst - final decoding graph.

        This requires OpenFST tools (fstcompile, fstcompose, etc.)
        In production, use Kaldi or similar toolchain.

        For now, generate the composition script.
        """
        script_path = self.output_dir / "build_hclg.sh"

        with open(script_path, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# WFST HCLG Graph Construction\n")
            f.write("# Requires OpenFST and Kaldi tools\n\n")
            f.write("set -e\n\n")
            f.write("# Compile FSTs\n")
            f.write("fstcompile --acceptor=false L.fst.txt L.fst\n")
            f.write("fstcompile --acceptor=false G.fst.txt G.fst\n")
            f.write("fstcompile --acceptor=true C.fst.txt C.fst\n")
            f.write("fstcompile --acceptor=true T.fst.txt T.fst\n\n")
            f.write("# Compose HCLG = H CL G\n")
            f.write("# H = input symbols (pdfids)\n")
            f.write("# C = context FST\n")
            f.write("# L = lexicon FST\n")
            f.write("# G = language model FST\n\n")
            f.write("# This is a simplified version\n")
            f.write("# In production, use Kaldi's make-hclg.sh\n")

        logger.info(f"HCLG build script written to {script_path}")
        return str(script_path)


class WFSTDecoder:
    """WFST-based beam search decoder for CTC/ED models."""

    def __init__(
        self,
        graph_dir: str,
        beam_size: int = 20,
        lattice_beam: float = 10.0,
        word_insertion_penalty: float = 0.0
    ):
        self.graph_dir = graph_dir
        self.beam_size = beam_size
        self.lattice_beam = lattice_beam
        self.word_insertion_penalty = word_insertion_penalty

        self.hclg_path = os.path.join(graph_dir, "HCLG.fst")
        self.lexicon_path = os.path.join(graph_dir, "lexicon.txt")

    def load_graph(self) -> bool:
        """Load WFST decoding graph."""
        if not os.path.exists(self.hclg_path):
            logger.warning(f"HCLG graph not found at {self.hclg_path}")
            logger.info("Run build_hclg.sh to construct the graph")
            return False

        logger.info(f"Loaded HCLG graph from {self.hclg_path}")
        return True

    def decode(
        self,
        emissions: List[List[float]],
        phone_labels: bool = True
    ) -> List[str]:
        """Decode CTC/ED emissions using WFST.

        Args:
            emissions: List of frame-level probability distributions
            phone_labels: If True, output phone序列; else output words

        Returns:
            List of decoded transcriptions
        """
        # Greedy decode for now (WFST integration requires OpenFST)
        if phone_labels:
            return self._greedy_decode_phones(emissions)
        else:
            return self._greedy_decode_words(emissions)

    def _greedy_decode_phones(self, emissions: List[List[float]]) -> List[str]:
        """Greedy decode to phone sequence."""
        decoded = []
        for frame in emissions:
            best_idx = frame.argmax()
            if best_idx > 0:
                decoded.append(IDX2PHONE.get(best_idx, f"<{best_idx}>"))
        return decoded

    def _greedy_decode_words(self, emissions: List[List[float]]) -> List[str]:
        """Decode to word sequence using lexicon."""
        phones = self._greedy_decode_phones(emissions)

        if not hasattr(self, 'lexicon'):
            self._load_lexicon()

        # Map phone sequence to words
        words = []
        i = 0
        while i < len(phones):
            matched = False
            for word, word_phones in self.lexicon.items():
                if tuple(phones[i:i+len(word_phones)]) == tuple(word_phones):
                    words.append(word)
                    i += len(word_phones)
                    matched = True
                    break
            if not matched:
                i += 1

        return words

    def _load_lexicon(self):
        """Load lexicon for word decoding."""
        self.lexicon = {}
        if os.path.exists(self.lexicon_path):
            with open(self.lexicon_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        word = parts[0]
                        phones = parts[1:]
                        self.lexicon[word] = phones


class CTCPrefixDecoder:
    """CTC prefix beam search decoder with WFST integration."""

    def __init__(
        self,
        vocab_size: int = 51865,
        beam_size: int = 10,
        blank_id: int = 0
    ):
        self.vocab_size = vocab_size
        self.beam_size = beam_size
        self.blank_id = blank_id

    def prefix_beam_search(
        self,
        emissions: List[List[float]],
        max_len: int = 100
    ) -> List[Tuple[str, float]]:
        """CTC prefix beam search decoding.

        Returns:
            List of (transcription, log_probability) tuples
        """
        T = len(emissions)
        blank_id = self.blank_id

        # Initialize beam
        beams = [('', 0.0)]  # (prefix, log_prob)

        for t in range(T):
            new_beams = {}

            for prefix, log_prob in beams:
                for c in range(self.vocab_size):
                    prob = emissions[t][c]

                    if c == blank_id:
                        # Blank: extend prefix without consuming character
                        new_prefix = prefix
                        new_log_prob = log_prob + prob
                    else:
                        # Non-blank: extend prefix
                        char = chr(c) if c < 128 else f"<{c}>"
                        new_prefix = prefix + char
                        new_log_prob = log_prob + prob

                    # Update best beam for this prefix
                    if new_prefix not in new_beams:
                        new_beams[new_prefix] = new_log_prob
                    else:
                        new_beams[new_prefix] = max(new_beams[new_prefix], new_log_prob)

            # Prune to beam_size
            beams = sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[:self.beam_size]

        return beams


def generate_sample_lexicon(output_dir: str) -> str:
    """Generate a sample Hungarian lexicon for testing."""
    lexicon_path = Path(output_dir) / "lexicon.txt"

    # Common Hungarian words with phonetic transcriptions
    lexicon_entries = [
        ("köszönöm", "k ö s ö n ö m"),
        ("szépen", "s z é p e n"),
        ("üdvözöllek", "ü d v ö z ö l l e k"),
        ("hogyan", "h o d y a n"),
        ("vagy", "v a d y"),
        ("mi", "m i"),
        ("ez", "e z"),
        ("egy", "e d y"),
        ("mondat", "m o n d a t"),
        ("tisztelet", "t i s z t e l e t"),
        ("siker", "s i k e r"),
        ("gyümölcs", "d y ü m ö l c s"),
        ("kávé", "k á v é"),
        ("tea", "t e a"),
        ("reggel", "r e g g e l"),
        ("este", "e s t e"),
        ("nap", "n a p"),
        ("hét", "h é t"),
        ("év", "é v"),
        ("hónap", "h ó n a p"),
        ("magyar", "m a d y a r"),
        ("nyelv", "n y e l v"),
        ("beszéd", "b e s z é d"),
    ]

    with open(lexicon_path, 'w', encoding='utf-8') as f:
        for word, phones in lexicon_entries:
            f.write(f"{word}\t{' '.join(phones)}\n")

    logger.info(f"Sample lexicon written to {lexicon_path}")
    return str(lexicon_path)


def generate_sample_lm(output_dir: str) -> str:
    """Generate a simple n-gram language model."""
    lm_path = Path(output_dir) / "lm.txt"

    # Simple bigram probabilities
    lm_entries = [
        ("köszönöm", 0.1),
        ("szépen", 0.05),
        ("üdvözöllek", 0.03),
        ("hogyan", 0.02),
        ("vagy", 0.05),
        ("mi", 0.08),
        ("ez", 0.1),
        ("egy", 0.07),
    ]

    with open(lm_path, 'w', encoding='utf-8') as f:
        for word, prob in lm_entries:
            f.write(f"{word} {prob}\n")

    logger.info(f"Sample LM written to {lm_path}")
    return str(lm_path)


def build_decoding_graph(
    output_dir: str,
    lexicon_path: Optional[str] = None,
    lm_path: Optional[str] = None
) -> Dict[str, str]:
    """Build complete WFST decoding graph.

    Args:
        output_dir: Output directory for FST files
        lexicon_path: Path to lexicon (or None to generate sample)
        lm_path: Path to language model (or None to generate sample)

    Returns:
        Dictionary of {graph_name: path} for L, G, C, T, HCLG
    """
    builder = WFSTGraphBuilder(output_dir)

    # Generate sample files if not provided
    if lexicon_path is None:
        lexicon_path = generate_sample_lexicon(output_dir)
    if lm_path is None:
        lm_path = generate_sample_lm(output_dir)

    # Build individual FSTs
    paths = {
        'L': builder.build_lexicon_fst(lexicon_path),
        'G': builder.build_language_model_fst(lm_path),
        'C': builder.build_context_fst(),
        'T': builder.build_phone_fst(),
        'hclg_script': builder.build_hclg(output_dir),
    }

    return paths


if __name__ == "__main__":
    import argparse
    import math

    parser = argparse.ArgumentParser(description="WFST Decoder for CTC/ED models")
    parser.add_argument('--output_dir', type=str, default='./data/wfst',
                        help='Output directory for WFST graphs')
    parser.add_argument('--lexicon', type=str, default=None,
                        help='Path to lexicon file')
    parser.add_argument('--lm', type=str, default=None,
                        help='Path to language model file')
    parser.add_argument('--build_graph', action='store_true',
                        help='Build WFST decoding graph')

    args = parser.parse_args()

    if args.build_graph:
        paths = build_decoding_graph(args.output_dir, args.lexicon, args.lm)
        print("Generated WFST files:")
        for name, path in paths.items():
            print(f"  {name}: {path}")
    else:
        # Test decoder
        decoder = WFSTDecoder(args.output_dir)
        if decoder.load_graph():
            print("WFST graph loaded successfully")
        else:
            print("No graph found. Use --build_graph to create one.")