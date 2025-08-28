#!/usr/bin/env python3
"""
Evaluate CRNN OCR across a small grid of decoding parameters.

Usage:
  python scripts/evaluate_crnn_params.py \
      --data-dir test_data \
      --service default \
      --lexicon configs/ocr/lexicon_words.txt \
      --lm configs/ocr/char_lm.json \
      --limit 0
"""
from __future__ import annotations
import argparse
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.append(str(PROJECT_ROOT))

from backend.application.services.ocr_service_crnn import OCRServiceWithCRNN
from backend.application.services.ocr_service_crnn_fixed import OCRServiceWithCRNNFixed


def extract_gt_from_filename(path: Path) -> str:
    """Extract ground truth from filename by keeping A-Z letters only in stem."""
    stem = path.stem
    # Keep only uppercase letters
    letters = re.findall(r"[A-Z]", stem)
    return "".join(letters)


def levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein distance (edit distance)."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ca = a[i - 1]
        for j in range(1, m + 1):
            cb = b[j - 1]
            cost = 0 if ca == cb else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )
    return dp[n][m]


@dataclass
class EvalResult:
    word_acc: float
    char_acc: float
    avg_time_ms: float
    total_samples: int


def evaluate_once(
    files: List[Path],
    service_kind: str,
) -> EvalResult:
    if service_kind == "fixed":
        service = OCRServiceWithCRNNFixed()
    else:
        service = OCRServiceWithCRNN()
    
    total_words = 0
    correct_words = 0
    total_chars = 0
    total_edit = 0
    times = []

    for img_path in files:
        gt = extract_gt_from_filename(img_path)
        if not gt:
            continue
        # Skip likely non-ground-truth files
        if gt.islower() or gt == "TEST":
            continue
        with open(img_path, 'rb') as f:
            data = f.read()
        t0 = time.time()
        res = service.process_image(data)
        dt = (time.time() - t0) * 1000.0
        times.append(dt)
        pred = ''.join([c.latin_equivalent for c in res.characters])
        # Normalize
        pred = re.sub(r"[^A-Z]", "", pred.upper())
        gt_n = re.sub(r"[^A-Z]", "", gt.upper())
        if not pred and not gt_n:
            continue
        total_words += 1
        if pred == gt_n:
            correct_words += 1
        # char accuracy via edit distance
        ed = levenshtein(pred, gt_n)
        total_edit += ed
        total_chars += len(gt_n)

    if total_words == 0 or total_chars == 0:
        return EvalResult(0.0, 0.0, float(np.mean(times)) if times else 0.0, 0)
    word_acc = correct_words / total_words
    char_acc = max(0.0, (total_chars - total_edit) / total_chars)
    avg_time = float(np.mean(times)) if times else 0.0
    return EvalResult(word_acc, char_acc, avg_time, total_words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=str, default='test_data')
    ap.add_argument('--glob', type=str, default='*.png')
    ap.add_argument('--service', type=str, default='default', choices=['default', 'fixed'])
    ap.add_argument('--lexicon', type=str, default='')
    ap.add_argument('--lm', type=str, default='')
    ap.add_argument('--limit', type=int, default=0, help='Limit number of files (0=all)')
    ap.add_argument('--grid', type=str, default='', help='Custom grid JSON (optional)')
    ap.add_argument('--ws-list', nargs='*', type=float, default=None, help='Width scale list for CRNN_WIDTH_SCALE (e.g., 1.0 1.08 1.15)')
    ap.add_argument('--rm-list', nargs='*', type=int, default=None, help='Right margin list for CRNN_RIGHT_MARGIN (e.g., 12 24)')
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob(args.glob))
    if args.limit and args.limit > 0:
        files = files[: args.limit]
    print(f"Evaluating on {len(files)} images from {data_dir}")

    # Base env
    base_env = os.environ.copy()
    # Optional lexicon/LM
    if args.lexicon:
        base_env['CRNN_LEXICON_PATH'] = str(Path(args.lexicon).resolve())
    if args.lm:
        base_env['CRNN_LM_PATH'] = str(Path(args.lm).resolve())
        base_env.setdefault('CRNN_LM_WEIGHT', '0.2')
    base_env.setdefault('CRNN_BEAM_SEARCH', 'true')

    # Default small grid
    base_grid = [
        {"CRNN_E_PENALTY": v1, "CRNN_RDN_BOOST": v2, "CRNN_BEAM_WIDTH": bw, "CRNN_LM_WEIGHT": lw}
        for v1 in [0.90, 0.95]
        for v2 in [1.00, 1.05]
        for bw in [3, 5]
        for lw in [0.0, 0.2]
    ]
    # Optional preproc grid
    ws_list = args.ws_list if args.ws_list else [None]
    rm_list = args.rm_list if args.rm_list else [None]
    grid = []
    for g in base_grid:
        for ws in ws_list:
            for rm in rm_list:
                cfg = dict(g)
                if ws is not None:
                    cfg["CRNN_WIDTH_SCALE"] = ws
                if rm is not None:
                    cfg["CRNN_RIGHT_MARGIN"] = rm
                grid.append(cfg)

    best: Tuple[float, Dict[str, float], EvalResult] | None = None
    for idx, cfg in enumerate(grid, 1):
        os.environ.clear()
        os.environ.update(base_env)
        for k, v in cfg.items():
            os.environ[k] = str(v)
        print(f"[{idx}/{len(grid)}] cfg={cfg}")
        res = evaluate_once(files, args.service)
        print(f"  => words={res.total_samples}, word_acc={res.word_acc:.3f}, char_acc={res.char_acc:.3f}, avg_ms={res.avg_time_ms:.1f}")
        score = (res.char_acc, res.word_acc)
        if best is None or score > (best[0], best[2].word_acc):
            best = (res.char_acc, cfg, res)

    if best:
        print("\nBest configuration:")
        print(best[1])
        br = best[2]
        print(f"word_acc={br.word_acc:.3f}, char_acc={br.char_acc:.3f}, avg_ms={br.avg_time_ms:.1f}, samples={br.total_samples}")
    else:
        print("No successful evaluations.")


if __name__ == '__main__':
    main()
