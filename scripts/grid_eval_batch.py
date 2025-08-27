#!/usr/bin/env python3
"""
Grid search decoding/preprocess hyperparameters across a dataset (default: test_data),
optionally building a lexicon from expected labels. Produces a summary and recommends
the best environment settings.

Usage:
  python scripts/grid_eval_batch.py [data_dir]

Outputs: results/grid_eval_batch/{summary.txt,recommended.env}
"""
from pathlib import Path
import os
import sys
import time
import numpy as np
import torch
import cv2
from PIL import Image

from backend.application.services.ocr_service_crnn_fixed import OCRServiceWithCRNNFixed


def parse_expected_from_filename(p: Path) -> str:
    stem = p.stem
    return stem.split('_')[0].replace('!', '').replace('.', '').upper()


def edit_distance(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    dp = [[0]*(lb+1) for _ in range(la+1)]
    for i in range(la+1):
        dp[i][0] = i
    for j in range(lb+1):
        dp[0][j] = j
    for i in range(1, la+1):
        for j in range(1, lb+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[la][lb]


def infer_once(service: OCRServiceWithCRNNFixed, img: np.ndarray) -> (str, float):
    pre = service._preprocess_for_crnn(img, target_height=64, max_width=service.max_width)
    tensor = torch.from_numpy(pre).unsqueeze(0).unsqueeze(0).float().to(service.device)
    with torch.no_grad():
        out = service.crnn_model(tensor)
        if service.use_beam:
            text, conf = service._ctc_beam_search_decode(out, service.beam_width)
        else:
            _, preds = out.max(2)
            text = service.crnn_converter.decode(preds)[0]
            probs = torch.exp(out)
            max_probs, _ = probs.max(2)
            cs = [max_probs[i, 0].item() for i, idx in enumerate(preds[:, 0]) if idx.item() != 0]
            conf = float(np.mean(cs)) if cs else 0.0
    return text, conf


def main():
    project = Path(__file__).resolve().parents[1]
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else (project / 'test_data')
    out_dir = project / 'results' / 'grid_eval_batch'
    out_dir.mkdir(parents=True, exist_ok=True)

    # collect dataset
    images = sorted([p for p in data_dir.glob('*.png') if '_' in p.stem])
    if not images:
        print(f"No images found in {data_dir}")
        return

    # lexicon from dataset expected labels
    lexicon_words = sorted({parse_expected_from_filename(p) for p in images if parse_expected_from_filename(p).isalpha()})
    lexicon_path = out_dir / 'lexicon.txt'
    with open(lexicon_path, 'w') as f:
        for w in lexicon_words:
            f.write(w + '\n')
    print(f"Lexicon built from data: {len(lexicon_words)} words")

    # search space (kept small to limit runtime)
    use_beam_list = [False, True]
    beam_width_list = [5]
    length_penalties = [1.0, 1.5]
    width_scales = [1.0, 1.03]
    right_margins = [12, 16]
    max_widths = [256]

    results = []

    for max_w in max_widths:
        for ws in width_scales:
            for rm in right_margins:
                for ub in use_beam_list:
                    for bw in beam_width_list:
                        for lp in length_penalties if ub else [1.0]:
                            # set env
                            os.environ['CRNN_BEAM_SEARCH'] = 'true' if ub else 'false'
                            os.environ['CRNN_BEAM_WIDTH'] = str(bw)
                            os.environ['CRNN_BEAM_LENGTH_PENALTY'] = str(lp)
                            os.environ['CRNN_WIDTH_SCALE'] = str(ws)
                            os.environ['CRNN_RIGHT_MARGIN'] = str(rm)
                            os.environ['CRNN_MAX_WIDTH'] = str(max_w)
                            os.environ['CRNN_LEXICON_PATH'] = str(lexicon_path)
                            os.environ['CRNN_LEXICON_STRICT'] = 'true'
                            os.environ['CRNN_LEXICON_BONUS'] = '1.5'

                            service = OCRServiceWithCRNNFixed()
                            service.use_beam = ub
                            service.beam_width = bw
                            service.width_scale = ws
                            service.right_margin = rm
                            service.max_width = max_w

                            total_dist = 0
                            total_words = 0
                            correct_words = 0
                            for p in images:
                                expected = parse_expected_from_filename(p)
                                img = np.array(Image.open(str(p)))
                                text, conf = infer_once(service, img)
                                d = edit_distance(text, expected)
                                total_dist += d
                                total_words += 1
                                if text == expected:
                                    correct_words += 1
                            acc = correct_words / total_words if total_words else 0.0
                            results.append({
                                'dist': total_dist,
                                'acc': acc,
                                'use_beam': ub,
                                'beam_width': bw,
                                'length_penalty': lp,
                                'width_scale': ws,
                                'right_margin': rm,
                                'max_width': max_w,
                            })

    # Sort best: minimal total edit distance, then higher acc
    results.sort(key=lambda r: (r['dist'], -r['acc']))
    best = results[0]

    # Print top-10
    print("\nTop 10 configs:")
    for i, r in enumerate(results[:10]):
        print(f"{i+1:2d}. dist={r['dist']:3d} acc={r['acc']*100:5.1f}% beam={r['use_beam']} bw={r['beam_width']} lp={r['length_penalty']} ws={r['width_scale']} rm={r['right_margin']} mw={r['max_width']}")

    # Save summary and recommended env
    with open(out_dir / 'summary.txt', 'w') as f:
        for r in results:
            f.write(f"dist={r['dist']:3d} acc={r['acc']*100:5.1f}% beam={r['use_beam']} bw={r['beam_width']} lp={r['length_penalty']} ws={r['width_scale']} rm={r['right_margin']} mw={r['max_width']}\n")
    with open(out_dir / 'recommended.env', 'w') as f:
        f.write(f"CRNN_LEXICON_PATH={lexicon_path}\n")
        f.write(f"CRNN_LEXICON_STRICT=true\n")
        f.write(f"CRNN_LEXICON_BONUS=1.5\n")
        f.write(f"CRNN_BEAM_SEARCH={'true' if best['use_beam'] else 'false'}\n")
        f.write(f"CRNN_BEAM_WIDTH={best['beam_width']}\n")
        f.write(f"CRNN_BEAM_LENGTH_PENALTY={best['length_penalty']}\n")
        f.write(f"CRNN_WIDTH_SCALE={best['width_scale']}\n")
        f.write(f"CRNN_RIGHT_MARGIN={best['right_margin']}\n")
        f.write(f"CRNN_MAX_WIDTH={best['max_width']}\n")
    print(f"\nSaved summary and recommended env to: {out_dir}")


if __name__ == '__main__':
    main()

