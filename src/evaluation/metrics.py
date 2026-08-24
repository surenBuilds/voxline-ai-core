"""
Evaluation metrics.

Provides metric functions for different benchmark categories.
Each metric takes actual output and expected/reference data,
returns a score between 0.0 and 1.0.

Normalization rules are documented in normalize.py.
"""

import re
from typing import Optional, List, Dict, Any
from difflib import SequenceMatcher

from src.evaluation.normalize import (
    normalize_for_comparison,
    normalize_numbers,
    normalize_case,
    normalize_whitespace,
)


def exact_match(actual: str, expected: str) -> float:
    """Exact string match after normalization."""
    return 1.0 if normalize_case(actual) == normalize_case(expected) else 0.0


def normalized_match(actual: str, expected: str, language: Optional[str] = None) -> float:
    """Normalized string match — applies full normalization pipeline."""
    a = normalize_for_comparison(actual, language)
    e = normalize_for_comparison(expected, language)
    return 1.0 if a == e else 0.0


def contains_match(actual: str, expected: str) -> float:
    """Check if expected substring is contained in actual."""
    return 1.0 if normalize_case(expected) in normalize_case(actual) else 0.0


def smart_contains(actual: str, expected: str, language: Optional[str] = None) -> float:
    """
    Normalized contains check — applies full normalization pipeline.
    Checks if normalized expected is a substring of normalized actual.
    """
    a = normalize_for_comparison(actual, language)
    e = normalize_for_comparison(expected, language)
    return 1.0 if e in a else 0.0


def any_contains(actual: str, references: List[str]) -> float:
    """Check if actual contains any of the reference strings."""
    if not references:
        return 0.0
    for ref in references:
        if normalize_case(ref) in normalize_case(actual):
            return 1.0
    return 0.0


def any_smart_contains(actual: str, references: List[str], language: Optional[str] = None) -> float:
    """Normalized contains check against multiple references."""
    if not references:
        return 0.0
    for ref in references:
        if smart_contains(actual, ref, language) >= 1.0:
            return 1.0
    return 0.0


def keyword_match(actual: str, keywords: List[str]) -> float:
    """Fraction of keywords found in actual response."""
    if not keywords:
        return 0.0
    actual_lower = normalize_case(actual)
    found = sum(1 for kw in keywords if normalize_case(kw) in actual_lower)
    return found / len(keywords)


def sequence_similarity(actual: str, expected: str) -> float:
    """SequenceMatcher ratio between actual and expected."""
    return SequenceMatcher(None, normalize_case(actual), normalize_case(expected)).ratio()


def best_reference_similarity(actual: str, references: List[str]) -> float:
    """Highest similarity score against all reference answers."""
    if not references:
        return 0.0
    return max(sequence_similarity(actual, ref) for ref in references)


def word_overlap(actual: str, expected: str) -> float:
    """Jaccard word overlap between actual and expected."""
    actual_words = set(normalize_case(actual).split())
    expected_words = set(normalize_case(expected).split())
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


# --- Task-Specific Metrics ---


def classification_accuracy(predicted: str, expected: str, valid_labels: List[str]) -> float:
    """Classification accuracy with normalized label matching."""
    def extract_label(text: str) -> str:
        text = normalize_case(text).strip()
        for label in valid_labels:
            if normalize_case(label) in text:
                return normalize_case(label)
        return text

    pred = extract_label(predicted)
    exp = extract_label(expected)
    return 1.0 if pred == exp else 0.0


def number_match(actual: str, expected: str) -> float:
    """Extract numbers from both strings and check if they match."""
    actual_nums = re.findall(r'-?\d+[\.,]?\d*', normalize_numbers(actual))
    expected_nums = re.findall(r'-?\d+[\.,]?\d*', normalize_numbers(expected))
    if not expected_nums:
        return 1.0 if not actual_nums else 0.0
    for en in expected_nums:
        for an in actual_nums:
            try:
                an_clean = an.replace(',', '')
                en_clean = en.replace(',', '')
                if abs(float(an_clean) - float(en_clean)) < 1e-6:
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


def vocabulary_accuracy(actual: str, expected: str, language: Optional[str] = None) -> float:
    """
    Vocabulary-focused metric: checks if the key term appears in the response.
    Uses smart_contains with normalized comparison.
    """
    return smart_contains(actual, expected, language)


def sentence_completion_match(actual: str, expected: str, language: Optional[str] = None) -> float:
    """
    Sentence completion metric: checks if the completion word/phrase appears.
    Combines smart_contains and best_reference_similarity.
    """
    contains_score = smart_contains(actual, expected, language)
    sim_score = best_reference_similarity(actual, [expected])
    return max(contains_score, sim_score)


def qa_match(actual: str, expected: str, references: Optional[List[str]] = None, language: Optional[str] = None) -> float:
    """
    Question answering metric: checks answer correctness.
    Priority: exact_match > normalized_match > number_match > any_smart_contains > best_reference_similarity
    """
    em = exact_match(actual, expected)
    if em >= 1.0:
        return 1.0

    nm = normalized_match(actual, expected, language)
    if nm >= 1.0:
        return 1.0

    # Check if both contain numbers that match
    if re.search(r'\d', expected):
        num = number_match(actual, expected)
        if num >= 1.0:
            return 1.0

    # Check contains with normalized comparison
    contains = smart_contains(actual, expected, language)
    if contains >= 1.0:
        return 1.0

    # Check against reference answers
    if references:
        for ref in references:
            if smart_contains(actual, ref, language) >= 1.0:
                return 1.0
            if number_match(actual, ref) >= 1.0:
                return 1.0

    # Fall back to similarity
    return best_reference_similarity(actual, references or [expected])


def instruction_following_score(actual: str, expected: str, instructions: Optional[List[str]] = None, language: Optional[str] = None) -> float:
    """
    Instruction following metric: checks if all constraints are met.
    Uses contains + similarity combined.
    """
    contains_score = smart_contains(actual, expected, language)
    sim_score = best_reference_similarity(actual, [expected])
    return max(contains_score, sim_score)


def translation_score(actual: str, expected: str, references: Optional[List[str]] = None, language: Optional[str] = None) -> float:
    """
    Translation metric: does NOT require one exact reference.
    Supports multiple reference translations.
    Uses best smart_contains and best_reference_similarity.
    """
    # Check against expected
    contains = smart_contains(actual, expected, language)
    sim = sequence_similarity(actual, expected)

    if contains >= 1.0 or sim >= 0.7:
        return 1.0

    # Check against references
    if references:
        for ref in references:
            if smart_contains(actual, ref, language) >= 1.0:
                return 1.0
            if sequence_similarity(actual, ref) >= 0.7:
                return 1.0

    return max(contains, sim)


def reasoning_score(actual: str, expected: str, references: Optional[List[str]] = None, language: Optional[str] = None) -> float:
    """
    Reasoning metric: checks if the logical answer is present.
    Combines number_match, smart_contains, and normalized_match.
    """
    em = exact_match(actual, expected)
    if em >= 1.0:
        return 1.0

    nm = normalized_match(actual, expected, language)
    if nm >= 1.0:
        return 1.0

    # Check number match for arithmetic
    if re.search(r'\d', expected):
        num = number_match(actual, expected)
        if num >= 1.0:
            return 1.0

    # Check contains
    contains = smart_contains(actual, expected, language)
    if contains >= 1.0:
        return 1.0

    # Check references
    if references:
        for ref in references:
            if smart_contains(actual, ref, language) >= 1.0:
                return 1.0
            if number_match(actual, ref) >= 1.0:
                return 1.0

    return max(contains, best_reference_similarity(actual, references or [expected]))


# --- Core computation ---


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
    language = config.get("language")

    if expected:
        metrics["exact_match"] = exact_match(actual, expected)
        metrics["normalized_match"] = normalized_match(actual, expected, language)
        metrics["contains"] = contains_match(actual, expected)
        metrics["smart_contains"] = smart_contains(actual, expected, language)
        metrics["sequence_similarity"] = sequence_similarity(actual, expected)
        metrics["word_overlap"] = word_overlap(actual, expected)

    if references:
        metrics["any_reference_match"] = any_contains(actual, references)
        metrics["any_smart_contains"] = any_smart_contains(actual, references, language)
        metrics["best_reference_similarity"] = best_reference_similarity(actual, references)

    # Number match
    if expected and re.search(r'\d', expected):
        metrics["number_match"] = number_match(actual, expected)

    # Task-specific metrics
    if category == "vocabulary" and expected:
        metrics["vocabulary_accuracy"] = vocabulary_accuracy(actual, expected, language)
    elif category == "sentence_completion" and expected:
        metrics["sentence_completion_match"] = sentence_completion_match(actual, expected, language)
    elif category == "question_answering" and expected:
        metrics["qa_match"] = qa_match(actual, expected, references, language)
    elif category == "instruction_following" and expected:
        metrics["instruction_following_score"] = instruction_following_score(actual, expected, references, language)
    elif category == "translation" and expected:
        metrics["translation_score"] = translation_score(actual, expected, references, language)
    elif category == "reasoning" and expected:
        metrics["reasoning_score"] = reasoning_score(actual, expected, references, language)
    elif category == "classification" and expected:
        valid_labels = config.get("valid_labels", [])
        if valid_labels:
            metrics["classification_accuracy"] = classification_accuracy(actual, expected, valid_labels)

    # Keyword match
    keywords = config.get("keywords", [])
    if keywords:
        metrics["keyword_match"] = keyword_match(actual, keywords)

    # Format check
    expected_format = config.get("expected_format")
    if expected_format:
        metrics["format_check"] = format_check(actual, expected_format)

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
