"""
Language detection and policy for Voxline AI.

LanguagePolicy is the single source of truth for language handling.
detect_language() provides lightweight Unicode-based detection.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Dict, Optional


class Language(Enum):
    """Supported languages."""
    ARMENIAN = "hy"
    ENGLISH = "en"
    UNKNOWN = "unknown"


_ARMENIAN_RANGE = re.compile(r"[\u0530-\u058F\uFB00-\uFB06\u2035\u2036\u2037\u2042]")
_LATIN_RANGE = re.compile(r"[A-Za-z]")
_CYRILLIC_RANGE = re.compile(r"[\u0400-\u04FF]")
_ARABIC_RANGE = re.compile(r"[\u0600-\u06FF]")


def detect_language(text: str) -> Language:
    """Detect the dominant language of a text using Unicode ranges.

    Returns Language.ARMENIAN when Armenian script is the dominant
    script among alphabetic characters. Returns Language.ENGLISH when
    Latin dominates. Returns Language.UNKNOWN otherwise.

    Mixed Armenian+English text is classified as ARMENIAN when
    Armenian characters are >= 30 % of total alphabetic characters.
    """
    if not text or not text.strip():
        return Language.UNKNOWN

    arm_count = len(_ARMENIAN_RANGE.findall(text))
    lat_count = len(_LATIN_RANGE.findall(text))
    total = arm_count + lat_count

    if total == 0:
        return Language.UNKNOWN

    arm_ratio = arm_count / total

    if arm_ratio >= 0.30:
        return Language.ARMENIAN
    if arm_ratio == 0 and lat_count > 0:
        return Language.ENGLISH
    if lat_count > arm_count:
        return Language.ENGLISH
    if arm_count > 0:
        return Language.ARMENIAN
    return Language.UNKNOWN


def is_armenian(text: str) -> bool:
    """Return True if the text is primarily Armenian."""
    return detect_language(text) == Language.ARMENIAN


def is_english(text: str) -> bool:
    """Return True if the text is primarily English."""
    return detect_language(text) == Language.ENGLISH


# ---------------------------------------------------------------------------
# Language Policy — single source of truth
# ---------------------------------------------------------------------------

_SYSTEM_INSTRUCTIONS: Dict[str, str] = {
    Language.ARMENIAN.value: (
        "You must answer entirely in natural Eastern Armenian. "
        "All sentences must be written in Armenian script. "
        "Do not answer in English or Russian unless the user explicitly requests it. "
        "Technical terms may remain in English when no Armenian equivalent exists, "
        "but the main explanation and sentence structure must be in Armenian."
    ),
    Language.ENGLISH.value: (
        "You are a helpful bilingual AI assistant. "
        "Be concise, accurate, and respectful. "
        "Reply in English."
    ),
}

_RETRY_INSTRUCTIONS: Dict[str, str] = {
    Language.ARMENIAN.value: (
        "The previous response violated the language policy. "
        "The user wrote in Armenian and you MUST answer in Armenian. "
        "Rewrite the answer entirely in natural Eastern Armenian. "
        "Do not explain the correction. Return only the corrected answer in Armenian."
    ),
}

# Temperature overrides for language compliance
LANGUAGE_TEMPERATURE: Dict[str, float] = {
    Language.ARMENIAN.value: 0.5,
    Language.ENGLISH.value: 0.7,
}


class LanguagePolicy:
    """Centralized language policy for Voxline.

    This is the single source of truth.  Do not duplicate language
    instructions in individual providers or assistant modules.
    """

    @staticmethod
    def get_system_instruction(lang: Language) -> str:
        return _SYSTEM_INSTRUCTIONS.get(lang.value, _SYSTEM_INSTRUCTIONS[Language.ENGLISH.value])

    @staticmethod
    def get_retry_instruction(lang: Language) -> str:
        return _RETRY_INSTRUCTIONS.get(lang.value, "")

    @staticmethod
    def get_temperature(lang: Language) -> Optional[float]:
        return LANGUAGE_TEMPERATURE.get(lang.value)

    @staticmethod
    def should_retry(user_lang: Language, response_text: Language) -> bool:
        """Return True when the response language mismatches the user language."""
        if user_lang == Language.UNKNOWN:
            return False
        if user_lang == Language.ARMENIAN and response_text == Language.ENGLISH:
            return True
        return False
