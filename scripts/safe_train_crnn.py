#!/usr/bin/env python3
"""
Safe wrapper to train CRNN without overwriting existing best model.
Saves with a timestamped filename and writes a recommendation file.

Usage:
  CRNN_EPOCHS=10 python scripts/safe_train_crnn.py [data_dir]
"""
from pathlib import Path
import shutil
import os
import datetime as dt

from scripts.train_crnn import train_crnn


def main():
    project = Path(__file__).resolve().parents[1]
    data_dir = Path(os.getenv('CRNN_DATA_DIR') or (project / 'test_data'))
    output_dir = project / 'models'
    output_dir.mkdir(exist_ok=True)

    ts = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    best_name = f'crnn_model_finetuned_{ts}.pth'

    # Backup existing canonical best if present
    canonical = output_dir / 'crnn_model_best.pth'
    if canonical.exists():
        backup = output_dir / f'backup_crnn_model_best_{ts}.pth'
        shutil.copy2(canonical, backup)
        print(f"Backed up existing best to {backup}")

    epochs = int(os.getenv('CRNN_EPOCHS', '10'))
    print(f"Training CRNN safely for {epochs} epochs. Output: {best_name}")
    model, history = train_crnn(data_dir, output_dir, epochs=epochs, best_model_name=best_name)

    # Write recommendation file with CRNN_MODEL_PATH
    candidate = output_dir / best_name
    recommend = output_dir / 'CRNN_MODEL_PATH.recommended'
    with open(recommend, 'w') as f:
        f.write(str(candidate))
    print(f"Candidate model: {candidate}")
    print(f"Set env to use it: CRNN_MODEL_PATH={candidate}")


if __name__ == '__main__':
    main()

