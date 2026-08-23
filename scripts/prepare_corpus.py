#!/usr/bin/env python3
"""Voxline AI Core — Corpus Preparation Script

Prepares training corpus for v0.4 language model training.
Loads raw text files, cleans them, splits into sentences,
and produces a formatted corpus file (one sentence per line).

Usage:
    python scripts/prepare_corpus.py --input data/raw --output data/voxline_corpus.txt
    python scripts/prepare_corpus.py --stats data/voxline_corpus.txt
"""

import argparse
import os
import re
import sys
import json
from pathlib import Path
from typing import List, Dict


def clean_text(text: str) -> str:
    """Clean raw text: normalize whitespace, remove artifacts."""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\S\n]+', ' ', text)
    return text


def split_sentences(text: str) -> List[str]:
    """Split text into sentences. Handles basic punctuation."""
    text = clean_text(text)
    parts = re.split(r'(?<=[.!?])\s+', text)
    sentences = []
    for part in parts:
        part = part.strip()
        if part and len(part) > 5:
            sentences.append(part)
    return sentences


def load_text_file(filepath: Path) -> str:
    """Load a text file with encoding detection."""
    encodings = ['utf-8', 'utf-8-sig', 'cp1257', 'cp1252', 'latin-1']
    for enc in encodings:
        try:
            return filepath.read_text(encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise UnicodeDecodeError(
        f"Cannot decode {filepath} with any known encoding"
    )


def load_line_by_line(filepath: Path) -> List[str]:
    """Load a file where each non-empty line is already a sentence."""
    content = load_text_file(filepath)
    sentences = []
    for line in content.splitlines():
        line = line.strip()
        if line and len(line) > 3:
            sentences.append(clean_text(line))
    return sentences


def is_line_by_line(content: str) -> bool:
    """Heuristic: if >70% of non-empty lines end without sentence punctuation, treat as line-by-line."""
    lines = [l.strip() for l in content.splitlines() if l.strip()]
    if not lines:
        return False
    no_punct = sum(1 for l in lines if not l.endswith(('.', '!', '?')))
    return no_punct / len(lines) > 0.7


def load_corpus(input_dir: Path) -> List[str]:
    """Load all text files from a directory."""
    all_sentences = []
    extensions = {'.txt', '.md', '.text'}

    for filepath in sorted(input_dir.rglob('*')):
        if filepath.suffix.lower() in extensions and filepath.is_file():
            print(f"  Loading: {filepath.name}")
            try:
                content = load_text_file(filepath)
                if is_line_by_line(content):
                    sentences = load_line_by_line(filepath)
                else:
                    sentences = split_sentences(content)
                all_sentences.extend(sentences)
                print(f"    -> {len(sentences)} sentences")
            except Exception as e:
                print(f"    WARNING: {e}")

    return all_sentences


def validate_sentence(sentence: str) -> bool:
    """Validate a sentence for training quality."""
    if len(sentence) < 3:
        return False
    if len(sentence) > 1000:
        return False
    if sentence.count(sentence[0]) == len(sentence):
        return False
    if re.match(r'^[\d\s\.\,\-\+\=]+$', sentence):
        return False
    return True


def compute_stats(sentences: List[str]) -> Dict:
    """Compute corpus statistics."""
    total_chars = sum(len(s) for s in sentences)
    total_words = sum(len(s.split()) for s in sentences)

    arm_chars = sum(1 for s in sentences for c in s if '\u0561' <= c <= '\u0587')
    eng_chars = sum(1 for s in sentences for c in s if 'a' <= c.lower() <= 'z')

    arm_lines = sum(1 for s in sentences if any('\u0561' <= c <= '\u0587' for c in s))
    eng_lines = sum(1 for s in sentences if any('a' <= c.lower() <= 'z' for c in s))

    return {
        'total_lines': len(sentences),
        'total_words': total_words,
        'total_chars': total_chars,
        'avg_words_per_line': total_words / max(len(sentences), 1),
        'armenian_chars': arm_chars,
        'english_chars': eng_chars,
        'armenian_lines': arm_lines,
        'english_lines': eng_lines,
        'char_distribution': {
            'armenian_pct': arm_chars / max(total_chars, 1) * 100,
            'english_pct': eng_chars / max(total_chars, 1) * 100,
        }
    }


def prepare_corpus(input_dir: Path, output_file: Path):
    """Main corpus preparation pipeline."""
    print(f"Loading corpus from: {input_dir}")
    sentences = load_corpus(input_dir)
    print(f"\nTotal loaded: {len(sentences)} sentences")

    print("Validating sentences...")
    valid = [s for s in sentences if validate_sentence(s)]
    invalid = len(sentences) - len(valid)
    print(f"Valid: {len(valid)}, Removed: {invalid}")

    seen = set()
    unique = []
    for s in valid:
        normalized = s.lower().strip()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(s)
    duplicates = len(valid) - len(unique)
    print(f"Unique: {len(unique)}, Duplicates removed: {duplicates}")

    unique.sort(key=lambda x: len(x))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in unique:
            f.write(sentence + '\n')

    print(f"\nSaved: {output_file}")
    print(f"Lines: {len(unique)}")

    stats = compute_stats(unique)
    print(f"\n--- Corpus Statistics ---")
    print(f"Total lines:      {stats['total_lines']}")
    print(f"Total words:      {stats['total_words']}")
    print(f"Total chars:      {stats['total_chars']}")
    print(f"Avg words/line:   {stats['avg_words_per_line']:.1f}")
    print(f"Armenian lines:   {stats['armenian_lines']} ({stats['char_distribution']['armenian_pct']:.1f}%)")
    print(f"English lines:    {stats['english_lines']} ({stats['char_distribution']['english_pct']:.1f}%)")

    stats_file = output_file.with_suffix('.stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Stats saved: {stats_file}")

    return stats


def show_stats(filepath: Path):
    """Show statistics for an existing corpus file."""
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]

    stats = compute_stats(sentences)
    print(f"\n--- Corpus Statistics: {filepath.name} ---")
    print(f"Total lines:      {stats['total_lines']}")
    print(f"Total words:      {stats['total_words']}")
    print(f"Total chars:      {stats['total_chars']}")
    print(f"Avg words/line:   {stats['avg_words_per_line']:.1f}")
    print(f"Armenian lines:   {stats['armenian_lines']} ({stats['char_distribution']['armenian_pct']:.1f}%)")
    print(f"English lines:    {stats['english_lines']} ({stats['char_distribution']['english_pct']:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Voxline AI Corpus Preparation'
    )
    parser.add_argument(
        '--input', '-i',
        type=Path,
        default=Path('data/raw'),
        help='Input directory with raw text files'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('data/voxline_corpus.txt'),
        help='Output corpus file'
    )
    parser.add_argument(
        '--stats', '-s',
        type=Path,
        default=None,
        help='Show stats for existing corpus file'
    )
    args = parser.parse_args()

    if args.stats:
        show_stats(args.stats)
    else:
        prepare_corpus(args.input, args.output)


if __name__ == '__main__':
    main()
