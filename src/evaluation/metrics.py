"""
Evaluation metrics.

Provides metric functions for different benchmark categories.
Each metric takes actual output and expected/reference data,
returns a score between 0.0 and 1.0.
"""

import re
from typing import Optional, List, Dict, Any
from difflib import SequenceMatcher


def exact_match(actual: str, expected: str) -> float:
    """Exact string match after normalization."""
    return 1.0 if actual.strip().lower() == expected.strip().lower() else 0.0


def contains_match(actual: str, expected: str) -> float:
    """Check if expected substring is contained in actual."""
    return 1.0 if expected.strip().lower() in actual.strip().lower() else 0.0


def any_contains(actual: str, references: List[str]) -> float:
    """Check if actual contains any of the reference strings."""
    if not references:
        return 0.0
    for ref in references:
        if ref.strip().lower() in actual.strip().lower():
            return 1.0
    return 0.0


def keyword_match(actual: str, keywords: List[str]) -> float:
    """Fraction of keywords found in actual response."""
    if not keywords:
        return 0.0
    actual_lower = actual.lower()
    found = sum(1 for kw in keywords if kw.lower() in actual_lower)
    return found / len(keywords)


def sequence_similarity(actual: str, expected: str) -> float:
    """SequenceMatcher ratio between actual and expected."""
    return SequenceMatcher(None, actual.strip().lower(), expected.strip().lower()).ratio()


def best_reference_similarity(actual: str, references: List[str]) -> float:
    """Highest similarity score against all reference answers."""
    if not references:
        return 0.0
    return max(sequence_similarity(actual, ref) for ref in references)


def word_overlap(actual: str, expected: str) -> float:
    """Jaccard word overlap between actual and expected."""
    actual_words = set(actual.lower().split())
    expected_words = set(expected.lower().split())
    if not expected_words:
        return 0.0
    intersection = actual_words & expected_words
    union = actual_words | expected_words
    return len(intersection) / len(union) if union else 0.0


def length_ratio(actual: str, expected: str) -> float:
    """Ratio of lengths, capped at 1.0. Penalizes very short or very long answers."""
    if not expected.strip():
        return 0.0
    ratio = len(actual.strip()) / len(expected.strip())
    return min(ratio, 1.0 / ratio) if ratio > 0 else 0.0


def classification_accuracy(predicted: str, expected: str, valid_labels: List[str]) -> float:
    """Classification accuracy with normalized label matching."""
    def extract_label(text: str) -> str:
        text = text.strip().lower()
        for label in valid_labels:
            if label.lower() in text:
                return label.lower()
        return text

    pred = extract_label(predicted)
    exp = extract_label(expected)
    return 1.0 if pred == exp else 0.0


def number_match(actual: str, expected: str) -> float:
    """Extract numbers from both strings and check if they match."""
    actual_nums = re.findall(r'-?\d+\.?\d*', actual)
    expected_nums = re.findall(r'-?\d+\.?\d*', expected)
    if not expected_nums:
        return 1.0 if not actual_nums else 0.0
    for en in expected_nums:
        for an in actual_nums:
            try:
                if abs(float(an) - float(en)) < 1e-6:
                    return 1.0
            except ValueError:
                pass
    return 0.0


def format_check(actual: str, expected_format: str) -> float:
    """Check if response follows expected format."""
    if expected_format == "numbered_list":
        lines = actual.strip().split('\n')
        numbered = sum(1 for l in lines if re.match(r'^\d+[\.\)]\s', l.strip()))
        return 1.0 if numbered >= 2 else 0.0
    elif expected_format == "bullet_list":
        lines = actual.strip().split('\n')
        bulleted = sum(1 for l in lines if re.match(r'^[-*•]\s', l.strip()))
        return 1.0 if bulleted >= 2 else 0.0
    elif expected_format == "short_answer":
        return 1.0 if len(actual.strip().split()) <= 20 else 0.0
    elif expected_format == "one_word":
        return 1.0 if len(actual.strip().split()) <= 2 else 0.0
    return 1.0


def context_retention_score(history: List[Dict[str, str]], response: str, key_phrase: str) -> float:
    """Check if response retains context from conversation history."""
    all_history = " ".join(m.get("content", "") for m in history)
    if key_phrase.lower() in all_history.lower():
        if any(word.lower() in response.lower() for word in key_phrase.split()):
            return 1.0
    return 0.0


def compute_case_metrics(
    actual: str,
    expected: Optional[str],
    references: Optional[List[str]] = None,
    category: str = "",
    metric_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Compute applicable metrics for a single case.

    Returns dict of metric_name -> score (0.0 to 1.0).
    """
    metrics: Dict[str, float] = {}
    config = metric_config or {}

    if expected:
        metrics["exact_match"] = exact_match(actual, expected)
        metrics["contains"] = contains_match(actual, expected)
        metrics["sequence_similarity"] = sequence_similarity(actual, expected)
        metrics["word_overlap"] = word_overlap(actual, expected)

    if references:
        metrics["any_reference_match"] = any_contains(actual, references)
        metrics["best_reference_similarity"] = best_reference_similarity(actual, references)

    keywords = config.get("keywords", [])
    if keywords:
        metrics["keyword_match"] = keyword_match(actual, keywords)

    valid_labels = config.get("valid_labels", [])
    if valid_labels and expected:
        metrics["classification_accuracy"] = classification_accuracy(actual, expected, valid_labels)

    expected_format = config.get("expected_format")
    if expected_format:
        metrics["format_check"] = format_check(actual, expected_format)

    if expected and re.search(r'\d', expected):
        metrics["number_match"] = number_match(actual, expected)

    return metrics


def aggregate_category_metrics(results: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate metrics across all cases in a category."""
    if not results:
        return {}
    agg: Dict[str, float] = {}
    all_keys = set()
    for r in results:
        all_keys.update(k for k in r if not k.startswith("_"))

    for key in all_keys:
        values = [r[key] for r in results if key in r]
        if values:
            agg[key] = sum(values) / len(values)
    return agg
