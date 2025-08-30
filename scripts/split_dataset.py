#!/usr/bin/env python3
"""
Split dataset images into train/eval without leakage by word.

Rules:
- Word label = uppercase A-Z letters from filename stem
- All images of the same word go to the same split
- Split by unique words using ratio (default 0.8 train)

Usage:
  python scripts/split_dataset.py \
    --src-dir test_data \
    --out-root train_eval_split \
    --ratio 0.8 \
    --seed 42
"""
from __future__ import annotations
import argparse
from pathlib import Path
import random
import re
import shutil


def word_from_stem(stem: str) -> str:
    return ''.join(re.findall(r"[A-Z]", stem.upper()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src-dir', type=str, default='test_data')
    ap.add_argument('--glob', type=str, default='*.png')
    ap.add_argument('--out-root', type=str, default='train_eval_split')
    ap.add_argument('--ratio', type=float, default=0.8, help='Train ratio by unique word')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src_dir)
    out_root = Path(args.out_root)
    train_dir = out_root / 'train'
    eval_dir = out_root / 'eval'
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob(args.glob))
    buckets: dict[str, list[Path]] = {}
    for p in files:
        w = word_from_stem(p.stem)
        if not w:
            continue
        buckets.setdefault(w, []).append(p)

    words = list(buckets.keys())
    random.Random(args.seed).shuffle(words)
    cutoff = int(len(words) * args.ratio)
    train_words = set(words[:cutoff])
    eval_words = set(words[cutoff:])

    n_train = n_eval = 0
    for w in train_words:
        for p in buckets[w]:
            dst = train_dir / p.name
            shutil.copy2(p, dst)
            n_train += 1
    for w in eval_words:
        for p in buckets[w]:
            dst = eval_dir / p.name
            shutil.copy2(p, dst)
            n_eval += 1

    print(f"Words: total={len(words)}, train={len(train_words)}, eval={len(eval_words)}")
    print(f"Files: train={n_train}, eval={n_eval}")
    print(f"Train dir: {train_dir}\nEval dir: {eval_dir}")


if __name__ == '__main__':
    main()

