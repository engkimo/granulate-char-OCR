#!/usr/bin/env python3
"""
Integrated evaluation:
- Generate lexicon from dataset filenames
- Run parameter grid evaluation with CRNN (strict lexicon ON)
- Save summaries and best-details/confusion under results/eval_<timestamp>

Usage:
  python scripts/run_eval_with_autolexicon.py \
    --data-dir test_data \
    --glob "*.png" \
    --ws-list 1.0 1.08 \
    --rm-list 12 24
"""
from __future__ import annotations
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


def run(cmd: list[str], env: dict | None = None):
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=str, default='test_data')
    ap.add_argument('--glob', type=str, default='*.png')
    ap.add_argument('--ws-list', nargs='*', type=str, default=['1.0','1.08'])
    ap.add_argument('--rm-list', nargs='*', type=str, default=['12','24'])
    args = ap.parse_args()

    out_dir = Path('results') / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    lex_path = Path('configs/ocr/lexicon_from_dataset.txt')
    # 1) generate lexicon
    run([
        'python', 'scripts/generate_lexicon_from_dataset.py',
        '--data-dir', args.data_dir,
        '--glob', args.glob,
        '--output', str(lex_path),
    ])

    # 2) evaluate with strict lexicon & preproc grid
    env = {
        'CRNN_LEXICON_STRICT': 'true'
    }
    run([
        'python', 'scripts/evaluate_crnn_params.py',
        '--data-dir', args.data_dir,
        '--service', 'fixed',
        '--lexicon', str(lex_path),
        '--lm', 'configs/ocr/char_lm.json',
        '--ws-list', *args.ws_list,
        '--rm-list', *args.rm_list,
        '--out-dir', str(out_dir),
    ], env=env)

    print(f"Done. Results written to: {out_dir}")


if __name__ == '__main__':
    main()
