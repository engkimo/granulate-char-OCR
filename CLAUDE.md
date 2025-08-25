# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Granulate character OCR system for recognizing fictional characters from Kamen Rider Gavv. The system converts Granulate characters to Latin alphabet (A-Z) equivalents using a multi-method recognition approach combining CRNN (end-to-end), CNN, Tesseract, and hash-based fallback methods.

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
# Data augmentation
python scripts/augment_data.py             # Generate augmented training data

# CNN model retraining with new data
python scripts/retrain_with_new_data.py

# CRNN model (end-to-end approach)
python scripts/train_crnn.py               # Train CRNN model
python scripts/evaluate_crnn.py            # Evaluate CRNN performance

# Evaluate model performance
python scripts/evaluate_new_testdata.py
python scripts/optimize_preprocessing.py   # Test preprocessing methods
```

## Architecture

### Clean Architecture Structure

The backend follows Clean Architecture with clear separation:

- **domain/**: Core business logic, no external dependencies
  - `entities/`: OCRResult, Character
- **application/**: Business rules and services
  - `services/ocr_service.py`: Main recognition logic
  - `services/ocr_service_improved.py`: Enhanced version with CRNN support
- **infrastructure/**: External integrations
  - `mapping/`: Granulate character mappings
- **api/**: FastAPI endpoints

### Recognition Pipeline

1. **Image Preprocessing**: 
   - Background detection and inversion
   - Bilateral filter for noise reduction
   - CLAHE for contrast enhancement
   - Binary thresholding
   - Morphological operations

2. **Recognition Methods** (in priority order):
   - **CRNN Model**: End-to-end text recognition without character segmentation
   - **Character Segmentation + CNN**: Horizontal projection analysis for boundaries
   - **Tesseract**: Secondary method, good for specific characters (L, P, R)
   - **Hash-based**: Fallback method

### Model Loading Strategy

```python
# Models are loaded in priority order:
crnn_path = project_root / 'models' / 'crnn_model_best.pth'          # Primary (end-to-end)
retrained_path = project_root / 'models' / 'cnn_model_retrained.pth'  # Secondary
original_path = project_root / 'models' / 'cnn_model_best.pth'        # Fallback
```

## Current Performance Metrics

### CRNN Model (Latest)
- **Word-level accuracy**: 61.6% (53/86 words)
- **Character-level accuracy**: 86.1% (297/345 characters)
- **Training**: 50 epochs, best validation accuracy 65.38%

### CNN Model (Character-based)
- **Test image accuracy**: 62.5% (5/8 characters on "PLEASURE")
- **New test data accuracy**: 9.1% character-level, 5.8% word-level
- **Processing time**: ~200ms per image

### Preprocessing Performance
- **Contrast Enhancement**: Best at 76.7% accuracy
- **Basic Processing**: Baseline at 73.3%
- **Adaptive Enhancement**: 70.0%

### Known Issues

- Common misrecognitions in CRNN: R→E, D→E, N→E
- Length prediction errors (e.g., 7 characters predicted as 8 or 15)
- Training/test data mismatch: Training used purple backgrounds with thick characters, test data has varied backgrounds with thin characters

## Critical Files and Locations

- **Test data**: `test_data/` - 86 real Granulate text images
- **Models**: `models/` - CNN and CRNN model files
- **Scripts**: `scripts/` - Training and evaluation scripts
- **Results**: `results/` - Evaluation outputs and visualizations
- **Training data**: `training_data/augmented/` - Original augmented data
- **Enhanced data**: `training_data/enhanced/` - Data with style variations

## Recent Architectural Changes

1. **CRNN Implementation**: 
   - VGG-like CNN feature extractor + Bidirectional LSTM
   - CTC loss for sequence learning
   - Supports variable-length text recognition without character segmentation

2. **Improved Preprocessing**: 
   - Adaptive preprocessing based on image characteristics
   - Color-aware processing for different image types
   - Text orientation correction

3. **Data Augmentation Pipeline**:
   - Character thickness variations (4 levels)
   - Background diversity (black, white, gradient, textured)
   - Realistic noise and lighting variations
   - 5 variations per original image

## Testing Specific Components

```bash
# Test hash mapping accuracy
python tests/debug/test_hash_mapping.py

# Analyze specific image
python tests/debug/analyze_pleasure_image.py

# Test integrated OCR with all methods
python tests/debug/test_integrated_ocr.py

# Test preprocessing methods
python preprocessing_results/granulate_preprocessor.py
```

## Environment Requirements

- Python 3.11+
- Tesseract OCR with custom `gran` language model at `/opt/homebrew/share/tessdata/gran.traineddata`
- PyTorch for ML models (required for CRNN)
- Node.js 18+ and pnpm for frontend

## Key Improvements Timeline

1. **Initial CNN model**: 12.5% → 62.5% accuracy on test image
2. **Real data evaluation**: Revealed 9.1% accuracy on new test data
3. **CRNN implementation**: Achieved 86.1% character accuracy, 61.6% word accuracy
4. **Data augmentation**: Created 150 variations per character for better generalization