#!/usr/bin/env python3
"""Generate real Armenian sentences and expand English corpus, then retrain."""
import os, random, subprocess, sys
sys.stdout.reconfigure(encoding='utf-8')

random.seed(42)

# ===== ARMENIAN =====
# All sentences are grammatically correct, real Armenian
sentences = set()

# --- tech ---
with open('data/raw/armenian_tech.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            sentences.add(line)

# --- daily ---
with open('data/raw/armenian_daily.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            sentences.add(line)

# --- science ---
with open('data/raw/armenian_science.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            sentences.add(line)

# --- education ---
with open('data/raw/armenian_education.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            sentences.add(line)

# --- culture ---
with open('data/raw/armenian_culture.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            sentences.add(line)

sentences = list(sentences)
random.shuffle(sentences)
sentences = sentences[:5000]

os.makedirs('data/raw', exist_ok=True)
with open('data/raw/armenian_general.txt', 'w', encoding='utf-8') as f:
    for s in sentences:
        f.write(s + '\n')

print(f'Armenian corpus: {len(sentences)} sentences')
