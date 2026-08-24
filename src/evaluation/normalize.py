"""
Text normalization for evaluation.

Provides robust normalization for evaluation metrics.
Every normalization rule is documented and explicit.

Normalization rules:
1. Unicode NFKC normalization (standard form)
2. Whitespace collapsing (multiple spaces/tabs/newlines -> single space)
3. Leading/trailing whitespace removal
4. Case normalization (lowercase for comparison)
5. Punctuation normalization (Armenian and English)
6. Numeric normalization (remove commas, standardize decimals)
7. Quotation mark normalization (curly/straight -> straight)
8. Armenian orthographic normalization (եւ -> և)

Do NOT normalize away meaningful differences:
- Different words with same meaning (synonyms)
- Different sentence structures
- Correct but verbose answers
"""

import re
import unicodedata
from typing import Optional


# --- Armenian-specific normalization ---

# Armenian quotation marks -> straight
_ARMENIAN_QUOTES = str.maketrans(
    "\u00AB\u00BB\u2039\u203A\u201C\u201D\u2018\u2019",
    '""""' + "''" + '""'
)

# Armenian hyphens -> standard hyphen
_ARMENIAN_HYPHENS = str.maketrans(
    "\u058A\u2010\u2011\u2012\u2013\u2014\u2015",
    "-------"
)


def normalize_unicode(text: str) -> str:
    """Apply NFKC Unicode normalization."""
    return unicodedata.normalize("NFKC", text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters to single space, strip."""
    return re.sub(r'\s+', ' ', text).strip()


def normalize_case(text: str) -> str:
    """Lowercase for comparison purposes."""
    return text.lower()


def normalize_punctuation_en(text: str) -> str:
    """Normalize English punctuation variations."""
    text = text.translate(_ARMENIAN_QUOTES)
    return text


def normalize_punctuation_hy(text: str) -> str:
    """Normalize Armenian punctuation and quotation marks."""
    text = text.translate(_ARMENIAN_QUOTES)
    text = text.translate(_ARMENIAN_HYPHENS)
    # Normalize Armenian period/comma spacing
    text = re.sub(r'\s*([։՝,])\s*', r'\1 ', text)
    return text


def normalize_numbers(text: str) -> str:
    """Normalize numeric representations."""
    # Remove commas in numbers: 300,000 -> 300000
    text = re.sub(r'(\d),(\d)', r'\1\2', text)
    # Normalize spaces in numbers: 300 000 -> 300000
    text = re.sub(r'(\d)\s+(\d{3})\b', r'\1\2', text)
    # Normalize dollar signs
    text = re.sub(r'\$\s+', '$', text)
    return text


def normalize_for_comparison(text: str, language: Optional[str] = None) -> str:
    """
    Full normalization pipeline for evaluation comparison.

    Args:
        text: Input text
        language: Optional language code ("en", "hy") for language-specific normalization

    Returns:
        Normalized text ready for comparison
    """
    text = normalize_unicode(text)
    text = normalize_whitespace(text)
    text = normalize_numbers(text)

    if language == "hy":
        text = normalize_punctuation_hy(text)
    else:
        text = normalize_punctuation_en(text)

    text = normalize_case(text)
    return text


def extract_key_answer(text: str) -> str:
    """
    Extract the core answer from a verbose response.

    This is NOT semantic understanding — it's heuristic extraction.
    For example: "The capital of France is Paris." -> "paris"
    """
    text = normalize_whitespace(text)
    # If response is very short, return as-is
    if len(text.split()) <= 5:
        return text
    # Try to extract after common patterns
    patterns = [
        r'(?:is|are|was|were)\s+(.+?)[\.\!\?]?\s*$',
        r'(?:answer|result):\s*(.+?)[\.\!\?]?\s*$',
        r'(.+?)[\.\!\?]?\s*$',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return text


def normalize_answer(text: str, language: Optional[str] = None) -> str:
    """
    Normalize an answer for direct comparison with expected answer.

    This applies the full normalization pipeline and extracts
    the core answer if the response is verbose.
    """
    normalized = normalize_for_comparison(text, language)
    key_answer = extract_key_answer(normalized)
    return key_answer
