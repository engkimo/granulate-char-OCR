#!/usr/bin/env python3
"""
Grid search CRNN decoding/preprocess hyperparameters on 20250826_test.png
and report best results against expected text "PLEASURE".

Search space (can be edited below):
  - use_beam: [False, True]
  - beam_width: [5]
  - beam_length_penalty: [1.0, 1.2, 1.5]
  - width_scale: [1.0, 1.02, 1.03, 1.05]
  - right_margin: [8, 12, 16, 24]
  - max_width: [256]

Outputs a markdown-like table and saves preprocessed image for best setting.
"""
from pathlib import Path
import os
import time
import numpy as np
import torch
import cv2
from PIL import Image

from backend.application.services.ocr_service_crnn_fixed import OCRServiceWithCRNNFixed


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


def run_once(service: OCRServiceWithCRNNFixed, img: np.ndarray) -> (str, float, float):
    # Use internal pipeline to respect instance/env settings
    start = time.time()
    pre = service._preprocess_for_crnn(img, target_height=64, max_width=service.max_width)
    # manual forward to avoid re-reading env
    tensor = torch.from_numpy(pre).unsqueeze(0).unsqueeze(0).float().to(service.device)
    with torch.no_grad():
        out = service.crnn_model(tensor)
        if service.use_beam:
            # pass env-controlled length penalty
            text, conf = service._ctc_beam_search_decode(out, service.beam_width)
        else:
            _, preds = out.max(2)
            text = service.crnn_converter.decode(preds)[0]
            probs = torch.exp(out)
            max_probs, _ = probs.max(2)
            cs = [max_probs[i, 0].item() for i, idx in enumerate(preds[:, 0]) if idx.item() != 0]
            conf = float(np.mean(cs)) if cs else 0.0
    elapsed = time.time() - start
    return text, conf, elapsed, pre


def main():
    project = Path(__file__).resolve().parents[1]
    image_path = project / "20250826_test.png"
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return
    expected = "PLEASURE"

    img = np.array(Image.open(str(image_path)))

    # search space
    use_beam_list = [False, True]
    beam_width_list = [5]
    length_penalties = [1.0, 1.2, 1.5]
    width_scales = [1.0, 1.02, 1.03, 1.05]
    right_margins = [8, 12, 16, 24]
    max_widths = [256]

    results = []

    for max_w in max_widths:
        for ws in width_scales:
            for rm in right_margins:
                for ub in use_beam_list:
                    for bw in beam_width_list:
                        for lp in length_penalties if ub else [1.0]:
                            # set env for functions that read env inside
                            os.environ["CRNN_BEAM_SEARCH"] = "true" if ub else "false"
                            os.environ["CRNN_BEAM_WIDTH"] = str(bw)
                            os.environ["CRNN_BEAM_LENGTH_PENALTY"] = str(lp)
                            os.environ["CRNN_WIDTH_SCALE"] = str(ws)
                            os.environ["CRNN_RIGHT_MARGIN"] = str(rm)
                            os.environ["CRNN_MAX_WIDTH"] = str(max_w)

                            service = OCRServiceWithCRNNFixed()
                            service.use_beam = ub
                            service.beam_width = bw
                            service.width_scale = ws
                            service.right_margin = rm
                            service.max_width = max_w

                            text, conf, t, pre = run_once(service, img)
                            dist = edit_distance(text, expected)
                            results.append({
                                'text': text,
                                'conf': conf,
                                'time_ms': t*1000,
                                'dist': dist,
                                'use_beam': ub,
                                'beam_width': bw,
                                'length_penalty': lp,
                                'width_scale': ws,
                                'right_margin': rm,
                                'max_width': max_w,
                                'pre': pre
                            })

    # sort: best edit distance, then higher confidence, then lower time
    results.sort(key=lambda r: (r['dist'], -r['conf'], r['time_ms']))

    # print top-10
    print("\nTop 10 configurations (by edit distance, -conf, time):")
    for i, r in enumerate(results[:10]):
        print(f"{i+1:2d}. dist={r['dist']} text={r['text']:<10} conf={r['conf']:.3f} time={r['time_ms']:.1f}ms "
              f"beam={r['use_beam']} bw={r['beam_width']} lp={r['length_penalty']} ws={r['width_scale']} rm={r['right_margin']} mw={r['max_width']}")

    # save best preprocessed image
    out_dir = project / "results" / "grid_eval_20250826"
    out_dir.mkdir(parents=True, exist_ok=True)
    best = results[0]
    pre = best['pre']
    pre_img = (np.clip(pre, 0.0, 1.0)*255).astype(np.uint8)
    cv2.imwrite(str(out_dir / "best_preprocessed.png"), pre_img)

    # save summary
    summary_path = out_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        for i, r in enumerate(results[:50]):
            f.write(f"{i+1:2d}. dist={r['dist']} text={r['text']:<12} conf={r['conf']:.3f} time={r['time_ms']:.1f}ms "
                    f"beam={r['use_beam']} bw={r['beam_width']} lp={r['length_penalty']} ws={r['width_scale']} rm={r['right_margin']} mw={r['max_width']}\n")
    print(f"\nSaved best preprocessed and summary to: {out_dir}")


if __name__ == "__main__":
    main()

