"""HTK format exporter for Hungarian speech datasets.

Exports datasets to HTK-compatible format:
- wav.scp: Audio file list (recording_id path)
- labels.mlf: Master Label File with transcriptions
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class HTKExporter:
    """Exports speech datasets to HTK format.

    Generates:
    - wav.scp: Script file mapping recording IDs to audio paths
    - labels.mlf: Master Label File with word-level transcriptions
    """

    def __init__(self, output_dir: str):
        """Initialize HTK exporter.

        Args:
            output_dir: Directory to write HTK files.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        data: List[Dict],
        wav_scp_filename: str = "wav.scp",
        labels_mlf_filename: str = "labels.mlf"
    ) -> Tuple[str, str]:
        """Export dataset to HTK format.

        Args:
            data: List of dicts with keys: 'id', 'audio_path', 'transcription'
            wav_scp_filename: Name of wav.scp file
            labels_mlf_filename: Name of labels.mlf file

        Returns:
            Tuple of (wav_scp_path, labels_mlf_path)
        """
        wav_scp_path = self.output_dir / wav_scp_filename
        labels_mlf_path = self.output_dir / labels_mlf_filename

        self._write_wav_scp(data, wav_scp_path)
        self._write_labels_mlf(data, labels_mlf_path)

        logger.info(f"Exported {len(data)} samples to {self.output_dir}")
        logger.info(f"  wav.scp: {wav_scp_path}")
        logger.info(f"  labels.mlf: {labels_mlf_path}")

        return str(wav_scp_path), str(labels_mlf_path)

    def _write_wav_scp(self, data: List[Dict], output_path: Path) -> None:
        """Write wav.scp file.

        Format: recording_id /full/path/to/audio.wav
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                rec_id = self._sanitize_id(item['id'])
                audio_path = os.path.abspath(item['audio_path'])
                f.write(f"{rec_id} {audio_path}\n")

    def _write_labels_mlf(self, data: List[Dict], output_path: Path) -> None:
        """Write labels.mlf (Master Label File).

        Format:
        #!MLF!#
        "*/labels.lab"
        word1
        word2
        ...
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("#!MLF!#\n")
            for item in data:
                rec_id = self._sanitize_id(item['id'])
                transcription = self._clean_transcription(item['transcription'])

                # MLF entry with quoted filename
                f.write(f"\"*/{rec_id}.lab\"\n")

                # Word-level labels (one per line)
                words = transcription.split()
                for word in words:
                    f.write(f"{word}\n")

                # End with blank line
                f.write("\n")

    def _sanitize_id(self, sample_id: str) -> str:
        """Sanitize sample ID for HTK compatibility.

        HTK IDs should be alphanumeric with underscores allowed.
        """
        # Replace problematic characters
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', str(sample_id))
        return sanitized

    def _clean_transcription(self, text: str) -> str:
        """Clean transcription text for HTK MLF.

        Remove extra whitespace, convert to lowercase.
        """
        if not text:
            return ""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Convert to lowercase
        text = text.lower()
        return text


class HTKParser:
    """Parser for reading HTK format files."""

    @staticmethod
    def parse_wav_scp(scp_path: str) -> Dict[str, str]:
        """Parse wav.scp file.

        Args:
            scp_path: Path to wav.scp file.

        Returns:
            Dict mapping recording_id -> audio_path
        """
        recordings = {}
        with open(scp_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(None, 1)
                if len(parts) == 2:
                    rec_id, audio_path = parts
                    recordings[rec_id] = audio_path
        return recordings

    @staticmethod
    def parse_labels_mlf(mlf_path: str) -> Dict[str, List[str]]:
        """Parse labels.mlf file.

        Args:
            mlf_path: Path to labels.mlf file.

        Returns:
            Dict mapping recording_id -> list of words
        """
        labels = {}
        current_id = None
        current_words = []

        with open(mlf_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                # Skip header
                if line.startswith('#') or line == "#!MLF!#":
                    continue

                # New entry starts with quoted filename
                if line.startswith('"'):
                    # Save previous entry
                    if current_id is not None and current_words:
                        labels[current_id] = current_words

                    # Extract ID from "*/id.lab"
                    match = re.search(r'\*/(.+)\.lab', line)
                    if match:
                        current_id = match.group(1)
                    else:
                        current_id = None
                    current_words = []
                elif line and current_id is not None:
                    # Word label
                    current_words.append(line)

        # Save last entry
        if current_id is not None and current_words:
            labels[current_id] = current_words

        return labels

    @staticmethod
    def get_aligned_data(
        wav_scp_path: str,
        labels_mlf_path: str
    ) -> List[Tuple[str, str, List[str]]]:
        """Get aligned audio-transcription pairs.

        Args:
            wav_scp_path: Path to wav.scp
            labels_mlf_path: Path to labels.mlf

        Returns:
            List of (recording_id, audio_path, words) tuples
        """
        recordings = HTKParser.parse_wav_scp(wav_scp_path)
        labels = HTKParser.parse_labels_mlf(labels_mlf_path)

        aligned = []
        for rec_id, audio_path in recordings.items():
            if rec_id in labels:
                aligned.append((rec_id, audio_path, labels[rec_id]))

        return aligned
