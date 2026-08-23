#!/usr/bin/env python3
"""Merge all English corpus parts and deduplicate."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

all_lines = []

files = [
    'data/raw/english_knowledge.txt',
    'data/raw/english_expand_part1.txt',
    'data/raw/english_expand_part2.txt',
    'data/raw/english_expand_part3.txt',
    'data/raw/english_expand_part4.txt',
    'data/raw/english_expand_part5.txt',
    'data/raw/english_expand_part6.txt',
    'data/raw/english_expand_part7.txt',
    'data/raw/english_expand_part8.txt',
]

for f in files:
    with open(f, 'r', encoding='utf-8') as fh:
        for l in fh:
            l = l.strip()
            if l and len(l) > 5:
                all_lines.append(l)

seen = set()
unique = []
for l in all_lines:
    n = l.lower().strip()
    if n not in seen:
        seen.add(n)
        unique.append(l)

with open('data/raw/english_knowledge.txt', 'w', encoding='utf-8') as f:
    for l in unique:
        f.write(l + '\n')

print(f"Final English corpus: {len(unique)} sentences")
