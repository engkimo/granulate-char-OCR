#!/usr/bin/env python3
"""
Generate pseudo-labeled samples from images using CRNN service.

For each input image, run CRNN-only inference and, if confidence >= threshold
and non-empty text, save the original image under output_dir as
  <PREDICTED_TEXT>_<index>.png

These outputs are directly consumable by scripts/train_crnn.py
because its dataset loader reads labels from filename stems.

Usage:
  python scripts/pseudo_label_from_images.py \
    --src-dir test_data \
    --out-dir training_data/pseudo \
    --min-conf 0.90 \
    --limit 0
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import cv2
from typing import Tuple

import os
from io import BytesIO
from PIL import Image

# Project imports
from backend.application.services.ocr_service_crnn_fixed import OCRServiceWithCRNNFixed


def crnn_only_predict(svc: OCRServiceWithCRNNFixed, image_bgr: np.ndarray) -> Tuple[str, float]:
    """Call the internal CRNN path to avoid classic fallback.
    Returns (text, confidence) or ('', 0.0) if fails.
    """
    try:
        # Use the protected method intentionally for CRNN-only
        res = svc._process_with_crnn(image_bgr)
        if res and res.get('text'):
            return res['text'], float(res.get('confidence', 0.0))
    except Exception:
        pass
    return '', 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src-dir', type=str, default='test_data')
    ap.add_argument('--glob', type=str, default='*.png')
    ap.add_argument('--out-dir', type=str, default='training_data/pseudo')
    ap.add_argument('--min-conf', type=float, default=0.9)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    src = Path(args.src_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Relax lexicon for broader coverage
    os.environ.setdefault('CRNN_LEXICON_STRICT', 'false')
    os.environ.setdefault('CRNN_BEAM_SEARCH', 'true')
    os.environ.setdefault('CRNN_BEAM_WIDTH', '5')

    svc = OCRServiceWithCRNNFixed()
    files = sorted(src.glob(args.glob))
    if args.limit and args.limit > 0:
        files = files[: args.limit]

    saved = 0
    for i, p in enumerate(files, 1):
        img = cv2.imread(str(p))
        if img is None:
            continue
        text, conf = crnn_only_predict(svc, img)
        text = ''.join([c for c in (text or '').upper() if c.isalpha()])
        if not text or conf < args.min_conf:
            continue
        dst = out / f"{text}_{i:04d}.png"
        cv2.imwrite(str(dst), img)
        saved += 1

    print(f"Saved {saved} pseudo-labeled samples to {out}")


if __name__ == '__main__':
    main()

