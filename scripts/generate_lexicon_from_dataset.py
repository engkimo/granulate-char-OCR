#!/usr/bin/env python3
"""
Generate a lexicon file (word list) from dataset filenames.

Rules:
- Extract only A-Z letters from each filename stem (before extension)
- Uppercase, deduplicate, sort
- Skip empty results

Usage:
  python scripts/generate_lexicon_from_dataset.py \
      --data-dir test_data \
      --glob "*.png" \
      --output configs/ocr/lexicon_from_dataset.txt
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path


def extract_word(stem: str) -> str:
    letters = re.findall(r"[A-Z]", stem.upper())
    return "".join(letters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=str, default='test_data')
    ap.add_argument('--glob', type=str, default='*.png')
    ap.add_argument('--output', type=str, default='configs/ocr/lexicon_from_dataset.txt')
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob(args.glob))
    words = set()
    for p in files:
        w = extract_word(p.stem)
        if w:
            words.add(w)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        for w in sorted(words):
            f.write(w + "\n")
    print(f"Wrote {len(words)} words to {out}")


if __name__ == '__main__':
    main()

