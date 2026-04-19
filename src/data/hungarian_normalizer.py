"""Hungarian text normalizer for Whisper fine-tuning pipeline."""

import re
from typing import Dict, Optional


class HungarianTextNormalizer:
    """Normalizes Hungarian text for Whisper training.

    Applies rigorous cleaning to ensure valid Hungarian characters
    and proper formatting for the tokenizer.
    """

    # Valid Hungarian lowercase characters (including special chars)
    HUNGARIAN_LOWERCASE = "aáäbcdeééfghiíjkloóöőpqrstuúüűvwxyz"
    # Valid Hungarian uppercase characters
    HUNGARIAN_UPPERCASE = "AÁÄBCDEÉÉFGHIÍJKLOÓÖŐPQRSTUÚÜŰVWXYZ"
    # Valid characters combined
    VALID_HUNGARIAN_CHARS = HUNGARIAN_LOWERCASE + HUNGARIAN_UPPERCASE

    # Valid punctuation to keep
    VALID_PUNCTUATION = ".,!?:-;'\"()"

    # Hungarian number words
    NUMBER_WORDS = {
        0: "nulla", 1: "egy", 2: "kettő", 3: "három", 4: "négy",
        5: "öt", 6: "hat", 7: "hét", 8: "nyolc", 9: "kilenc",
        10: "tíz", 11: "tizenegy", 12: "tizenkettő", 13: "tizenhárom",
        14: "tizennégy", 15: "tizenöt", 16: "tizenhat", 17: "tizenhét",
        18: "tizennyolc", 19: "tizenkilenc", 20: "húsz", 30: "harminc",
        40: "negyven", 50: "ötven", 60: "hatvan", 70: "hetven",
        80: "nyolcvan", 90: "kilencven", 100: "száz", 1000: "ezer"
    }

    # Common Hungarian abbreviations
    ABBREVIATIONS = {
        "kb.": "körülbelül",
        "stb.": "és így tovább",
        "stb": "és így tovább",
        "pl.": "például",
        "pl": "például",
        "st.": "szent",
        "dr.": "doktor",
        "dr": "doktor",
        "mln.": "millió",
        "mrd.": "milliárd",
        "kg.": "kilogramm",
        "km.": "kilométer",
        "cm.": "centiméter",
        "mm.": "milliméter",
        "°C": "celsius fok",
        "%": "százalék"
    }

    def __init__(self, remove_abbreviations: bool = True, expand_numbers: bool = True):
        """Initialize the normalizer.

        Args:
            remove_abbreviations: Whether to expand Hungarian abbreviations.
            expand_numbers: Whether to expand numbers to Hungarian words.
        """
        self.remove_abbreviations = remove_abbreviations
        self.expand_numbers = expand_numbers

    def normalize(self, text: str) -> str:
        """Apply full normalization pipeline to Hungarian text.

        Args:
            text: Input text to normalize.

        Returns:
            Normalized Hungarian text.
        """
        if not text or not isinstance(text, str):
            return ""

        # Step 1: Basic whitespace normalization
        text = re.sub(r'\s+', ' ', text.strip())

        # Step 2: Handle abbreviations
        if self.remove_abbreviations:
            text = self._expand_abbreviations(text)

        # Step 3: Handle numbers
        if self.expand_numbers:
            text = self._expand_numbers(text)

        # Step 4: Remove invalid punctuation
        text = self._filter_punctuation(text)

        # Step 5: Remove non-Hungarian characters (except spaces and kept punctuation)
        text = self._filter_characters(text)

        # Step 6: Final whitespace cleanup
        text = re.sub(r'\s+', ' ', text.strip())

        return text.lower()

    def _expand_abbreviations(self, text: str) -> str:
        """Expand Hungarian abbreviations."""
        for abbrev, expansion in self.ABBREVIATIONS.items():
            # Use word boundary matching
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            text = re.sub(pattern, expansion, text)
        return text

    def _expand_numbers(self, text: str) -> str:
        """Convert numbers to Hungarian words.

        Handles numbers 0-9999, years, and simple decimals.
        """
        def _number_to_words(match):
            num_str = match.group()
            try:
                num = int(num_str)
                return self._number_to_hungarian(num)
            except ValueError:
                return num_str

        # Match numbers (including years like 2024, 1990)
        text = re.sub(r'\b\d+(?:[.,]\d+)?\b', _number_to_words, text)
        return text

    def _number_to_hungarian(self, num: int) -> str:
        """Convert an integer to Hungarian words."""
        if num in self.NUMBER_WORDS:
            return self.NUMBER_WORDS[num]

        if num < 0:
            return "mínusz " + self._number_to_hungarian(abs(num))

        if num < 100:
            tens = (num // 10) * 10
            ones = num % 10
            result = self.NUMBER_WORDS.get(tens, "")
            if ones > 0:
                result += self.NUMBER_WORDS.get(ones, "")
            return result

        if num < 1000:
            hundreds = num // 100
            remainder = num % 100
            result = self.NUMBER_WORDS.get(hundreds, str(hundreds)) + "száz"
            if remainder > 0:
                result += self._number_to_hungarian(remainder)
            return result

        if num < 1000000:
            thousands = num // 1000
            remainder = num % 1000
            if thousands == 1:
                result = "ezer"
            else:
                result = self._number_to_hungarian(thousands) + "ezer"
            if remainder > 0:
                result += self._number_to_hungarian(remainder)
            return result

        return str(num)

    def _filter_punctuation(self, text: str) -> str:
        """Keep only valid punctuation marks."""
        filtered = []
        for char in text:
            if char in self.VALID_HUNGARIAN_CHARS or char == ' ' or char.isdigit():
                filtered.append(char)
            elif char in self.VALID_PUNCTUATION:
                filtered.append(char)
            elif char in '.,' and len(filtered) > 0 and filtered[-1].isdigit():
                # Keep decimal separators in numbers
                filtered.append(char)
            else:
                filtered.append(' ')
        return ''.join(filtered)

    def _filter_characters(self, text: str) -> str:
        """Remove any characters not in valid Hungarian set."""
        filtered = []
        for char in text:
            if char in self.VALID_HUNGARIAN_CHARS or char in self.VALID_PUNCTUATION or char == ' ':
                filtered.append(char)
        return ''.join(filtered)

    def is_valid_transcription(self, text: str) -> bool:
        """Check if transcription meets quality thresholds.

        Args:
            text: Normalized text to validate.

        Returns:
            True if text is valid for training.
        """
        if not text:
            return False

        # Must have at least some Hungarian characters
        has_hungarian = any(c in self.VALID_HUNGARIAN_CHARS for c in text)
        if not has_hungarian:
            return False

        # Should not be all numbers
        if text.replace(' ', '').replace('.', '').isdigit():
            return False

        # Minimum length check (at least 2 words)
        words = text.split()
        if len(words) < 2:
            return False

        return True


def normalize_batch(texts: list, normalizer: Optional[HungarianTextNormalizer] = None) -> list:
    """Normalize a batch of texts.

    Args:
        texts: List of input texts.
        normalizer: HungarianTextNormalizer instance (creates default if None).

    Returns:
        List of normalized texts.
    """
    if normalizer is None:
        normalizer = HungarianTextNormalizer()

    return [normalizer.normalize(text) for text in texts]
