"""Custom dataset loader for HTK format data.

Reads from wav.scp and labels.mlf files and provides
speech samples for Whisper training.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import librosa
import logging

logger = logging.getLogger(__name__)


class HTKHungarianDataset:
    """Dataset loaded from HTK format files.

    Compatible with HuggingFace training pipeline.

    Args:
        wav_scp_path: Path to wav.scp file.
        labels_mlf_path: Path to labels.mlf file.
        feature_extractor: WhisperFeatureExtractor for audio processing.
        tokenizer: WhisperTokenizer for text processing.
        audio_dir: Optional base directory for audio files (if relative paths in scp).
        max_audio_length: Maximum audio duration in seconds.
        min_audio_length: Minimum audio duration in seconds.
    """

    def __init__(
        self,
        wav_scp_path: str,
        labels_mlf_path: str,
        feature_extractor,
        tokenizer,
        audio_dir: Optional[str] = None,
        max_audio_length: float = 30.0,
        min_audio_length: float = 1.0
    ):
        from .htk_exporter import HTKParser

        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.audio_dir = audio_dir
        self.max_audio_length = max_audio_length
        self.min_audio_length = min_audio_length

        # Parse HTK files
        recordings = HTKParser.parse_wav_scp(wav_scp_path)
        labels = HTKParser.parse_labels_mlf(labels_mlf_path)

        # Build aligned dataset
        self.samples = []
        for rec_id, audio_path in recordings.items():
            if rec_id not in labels:
                logger.warning(f"Missing transcription for {rec_id}, skipping")
                continue

            # Resolve audio path
            if not os.path.isabs(audio_path) and audio_dir:
                audio_path = os.path.join(audio_dir, audio_path)

            self.samples.append({
                'id': rec_id,
                'audio_path': audio_path,
                'transcription': ' '.join(labels[rec_id])
            })

        logger.info(f"Loaded {len(self.samples)} samples from HTK files")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Dict with 'input_features' and 'labels' for Whisper.
        """
        sample = self.samples[idx]

        # Load audio
        audio_path = sample['audio_path']
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
        except Exception as e:
            logger.error(f"Failed to load audio {audio_path}: {e}")
            # Return empty sample on error
            audio = np.zeros(int(16000 * self.min_audio_length), dtype=np.float32)
            sr = 16000

        # Check duration
        duration = len(audio) / sr
        if duration < self.min_audio_length or duration > self.max_audio_length:
            # Pad or trim
            if duration < self.min_audio_length:
                audio = np.pad(audio, (0, int(16000 * (self.min_audio_length - duration))))
            else:
                audio = audio[:int(16000 * self.max_audio_length)]

        # Extract features
        input_features = self.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features[0]

        # Tokenize transcription
        labels = self.tokenizer(
            sample['transcription'],
            return_tensors="pt"
        ).input_ids[0]

        return {
            'input_features': input_features,
            'labels': labels,
            'id': sample['id']
        }


def create_htk_dataloader(
    wav_scp_path: str,
    labels_mlf_path: str,
    feature_extractor,
    tokenizer,
    batch_size: int = 4,
    shuffle: bool = True,
    audio_dir: Optional[str] = None,
    max_audio_length: float = 30.0,
    min_audio_length: float = 1.0
) -> 'HTKDataLoader':
    """Create a DataLoader from HTK files.

    Returns a DataLoader wrapping HTKHungarianDataset.
    """
    from torch.utils.data import DataLoader
    from .collator import HTKDataCollator

    dataset = HTKHungarianDataset(
        wav_scp_path=wav_scp_path,
        labels_mlf_path=labels_mlf_path,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        audio_dir=audio_dir,
        max_audio_length=max_audio_length,
        min_audio_length=min_audio_length
    )

    collator = HTKDataCollator(tokenizer=tokenizer)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=0
    )

    return dataloader
