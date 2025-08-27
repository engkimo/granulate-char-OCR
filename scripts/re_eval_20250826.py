#!/usr/bin/env python3
"""
Re-evaluate a single image with CRNN (fixed service),
save preprocessed images for different max_width settings,
and print recognized text with average confidence.

Usage:
  python scripts/re_eval_20250826.py [image_path]

Outputs are saved under results/re_eval_20250826_*/
"""
import sys
from pathlib import Path
import time
import numpy as np
import torch
import cv2
from PIL import Image

from backend.application.services.ocr_service_crnn_fixed import OCRServiceWithCRNNFixed


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_image(path: Path, arr: np.ndarray):
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    cv2.imwrite(str(path), arr)


def greedy_decode_and_confidence(model, converter, img_tensor: torch.Tensor):
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)  # (seq_len, batch, num_classes), log_softmax
        # greedy indices
        _, preds = output.max(2)
        # convert to probabilities
        probs = torch.exp(output)  # since output is log_softmax
        max_probs, _ = probs.max(2)

        # decode CTC
        text = converter.decode(preds)[0]

        # compute avg confidence for non-blank positions
        conf_scores = []
        for i, idx in enumerate(preds[:, 0]):
            if idx.item() != 0:  # 0 is blank
                conf_scores.append(max_probs[i, 0].item())
        avg_conf = float(np.mean(conf_scores)) if conf_scores else 0.0

        return text, avg_conf


def main():
    project_root = Path(__file__).resolve().parents[1]
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else project_root / "20250826_test.png"
    out_dir = project_root / "results" / "re_eval_20250826"
    ensure_dir(out_dir)

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    # Load image
    pil_img = Image.open(str(image_path))
    image_np = np.array(pil_img)

    service = OCRServiceWithCRNNFixed()

    # Evaluate via full pipeline (default max_width inside service)
    print("\n== Full pipeline (service.process_image) ==")
    start = time.time()
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    result = service.process_image(image_bytes)
    elapsed = time.time() - start
    text = ''.join([c.latin_equivalent for c in result.characters])
    avg_conf = result.average_confidence if hasattr(result, 'average_confidence') else 0.0
    print(f"Text: {text}")
    print(f"Avg confidence (from result): {avg_conf:.3f}")
    print(f"Processing time: {elapsed*1000:.1f} ms")

    # Evaluate with custom max_width settings using internal preprocessing
    widths = [256, 320, 384]
    for w in widths:
        pre = service._preprocess_for_crnn(image_np, target_height=64, max_width=w)
        save_image(out_dir / f"preprocessed_w{w}.png", pre)

        # forward pass manually to keep max_width override
        img_tensor = torch.from_numpy(pre).unsqueeze(0).unsqueeze(0).float()
        img_tensor = img_tensor.to(service.device)
        text_w, conf_w = greedy_decode_and_confidence(service.crnn_model, service.crnn_converter, img_tensor)
        print(f"\n-- CRNN greedy (max_width={w}) --")
        print(f"Text: {text_w}")
        print(f"Avg confidence: {conf_w:.3f}")

    print(f"\nSaved preprocessed images to: {out_dir}")


if __name__ == "__main__":
    main()

