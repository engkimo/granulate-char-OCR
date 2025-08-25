# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Granulate character OCR system for recognizing fictional characters from Kamen Rider Gavv. The system converts Granulate characters to Latin alphabet (A-Z) equivalents using a multi-method recognition approach combining CNN, Tesseract, and hash-based fallback methods.

## Development Commands

### Backend (Python/FastAPI)

```bash
# Setup and run
uv venv
uv pip install -e ".[dev]"
uv run uvicorn backend.main:app --reload

# Testing
uv run pytest                              # Run all tests
uv run pytest backend/tests/unit/          # Unit tests only
uv run pytest --cov=backend                # With coverage

# Code quality
uv run black backend/                      # Format code
uv run flake8 backend/                     # Lint code
uv run mypy backend/                       # Type checking
```

### Frontend (React/TypeScript)

```bash
cd front
pnpm install
pnpm run dev                               # Start dev server
pnpm run build                             # Build for production
pnpm test                                  # Run tests with Vitest
pnpm run typecheck                         # TypeScript checking
pnpm run deploy                            # Deploy to Cloudflare Workers
```

### Model Training and Evaluation

```bash
# CNN model retraining with new data
python scripts/retrain_with_new_data.py

# Evaluate model performance
python scripts/evaluate_new_testdata.py

# CRNN model (end-to-end approach)
python scripts/train_crnn.py               # Train CRNN model
python scripts/evaluate_crnn.py            # Evaluate CRNN performance
```

## Architecture

### Clean Architecture Structure

The backend follows Clean Architecture with clear separation:

- **domain/**: Core business logic, no external dependencies
  - `entities/`: OCRResult, Character
- **application/**: Business rules and services
  - `services/ocr_service.py`: Main recognition logic
- **infrastructure/**: External integrations
  - `mapping/`: Granulate character mappings
- **api/**: FastAPI endpoints

### Recognition Pipeline

1. **Image Preprocessing**: Background detection, noise reduction, contrast enhancement, binary thresholding
2. **Character Segmentation**: Horizontal projection analysis for character boundary detection
3. **Multi-Method Recognition**:
   - CNN Model: Primary method with confidence scoring
   - Tesseract: Secondary, good for specific characters (L, P, R)
   - Hash-based: Fallback method

### Model Loading Strategy

```python
# Models are loaded in priority order:
retrained_path = project_root / 'models' / 'cnn_model_retrained.pth'  # First choice
original_path = project_root / 'models' / 'cnn_model_best.pth'        # Fallback
```

## Current Performance Metrics

- **Test image accuracy**: 62.5% (5/8 characters on "PLEASURE")
- **New test data accuracy**: 9.1% character-level, 5.8% word-level
- **Processing time**: ~200ms per image

### Known Issues

- Training/test data mismatch: Training used purple backgrounds with thick characters, test data has varied backgrounds with thin characters
- Common misrecognitions: E→C, O→L, O→C

## Critical Files and Locations

- **Test data**: `test_data/` - 86 real Granulate text images
- **Models**: `models/` - CNN and CRNN model files
- **Scripts**: `scripts/` - Training and evaluation scripts
- **Results**: `results/` - Evaluation outputs and visualizations

## Recent Architectural Changes

1. **CRNN Implementation**: Added end-to-end text recognition model that doesn't require character segmentation
2. **Improved Preprocessing**: Adaptive preprocessing based on image characteristics
3. **Model Retraining**: Support for retraining with real Granulate text images

## Testing Specific Components

```bash
# Test hash mapping accuracy
python tests/debug/test_hash_mapping.py

# Analyze specific image
python tests/debug/analyze_pleasure_image.py

# Test integrated OCR with all methods
python tests/debug/test_integrated_ocr.py
```

## Environment Requirements

- Python 3.11+
- Tesseract OCR with custom `gran` language model at `/opt/homebrew/share/tessdata/gran.traineddata`
- PyTorch for ML models (optional but recommended)
- Node.js 18+ and pnpm for frontend