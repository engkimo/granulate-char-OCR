# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Granulate character OCR system for recognizing fictional characters from Kamen Rider Gavv. The system converts Granulate characters to their corresponding Latin alphabet (A-Z) and numbers (0-9) equivalents.

## Development Commands

### Backend (Python/FastAPI)

```bash
# Initial setup (using uv package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh  # Install uv if needed
uv venv                                           # Create virtual environment
uv pip install -e ".[dev]"                        # Install with dev dependencies
uv run uvicorn backend.main:app --reload          # Start development server on :8000

# Testing
uv run pytest                                     # Run all tests
uv run pytest backend/tests/unit/                 # Run unit tests only
uv run pytest backend/tests/integration/          # Run integration tests only
uv run pytest -v -k "test_ocr"                   # Run specific test by name pattern
uv run pytest --cov=backend                       # Run with coverage

# Code quality
uv run black backend/                             # Format code
uv run flake8 backend/                           # Lint code
uv run mypy backend/                             # Type checking
```

### Frontend (React/TypeScript)

```bash
# Initial setup (using pnpm in front/ directory)
cd front
pnpm install                                     # Install dependencies

# Development
pnpm run dev                                     # Start dev server with hot reload
pnpm run build                                   # Build for production
pnpm run preview                                 # Preview production build locally

# Testing
pnpm test                                        # Run all tests with vitest
pnpm test:ui                                     # Run tests with UI
pnpm test:coverage                               # Run tests with coverage

# Code quality
pnpm run typecheck                               # Run TypeScript type checking

# Deployment (Cloudflare Workers)
pnpm run deploy                                  # Deploy to Cloudflare Workers
pnpm run cf-typegen                              # Generate Cloudflare types
```

### OCR Testing Scripts

```bash
# Test real image recognition (from project root)
python tests/integration/test_real_image.py      # Test with PLEASURE image

# Test API integration (requires server running)
python tests/integration/test_ocr_api.py         # Test all characters A-Z

# Debug and analysis scripts
python tests/debug/test_hash_mapping.py          # Test hash-based recognition (28.5% accuracy)
python tests/debug/test_similarity_mapping.py    # Test similarity-based recognition
python tests/debug/analyze_pleasure_image.py     # Analyze PLEASURE test image
```

## Architecture

### Backend Structure (Clean Architecture)

The backend follows Clean Architecture principles with clear separation of concerns:

- **domain/**: Core business logic with no external dependencies
  - `entities/`: Core business objects (OCRResult, Character)
  - `repositories/`: Abstract interfaces for data access
  
- **application/**: Use cases and business rules
  - `services/OCRService`: Main service implementing recognition logic
  - `use_cases/`: Specific business operations
  
- **infrastructure/**: External dependencies and implementations
  - `ocr/`: Tesseract and ML model integrations
  - `mapping/`: Granulate character hash mappings
  
- **api/**: FastAPI routes and DTOs
  - `routers/`: HTTP endpoints
  - `models/`: Request/Response schemas

### Frontend Structure (React Router v7)

Modern React application with Cloudflare Workers deployment:

- **app/routes/**: Route components following React Router v7 conventions
- **app/components/Camera/**: Enhanced camera component with real-time processing
- **app/services/**: API client and external service integrations
- **app/stores/**: Zustand state management

### Recognition Pipeline

1. **Image Preprocessing** (`_preprocess_image`)
   - Grayscale conversion
   - Background detection and inversion if needed
   - Bilateral filter for noise reduction
   - CLAHE for contrast enhancement
   - Binary thresholding
   - Morphological operations

2. **Character Segmentation** (`_extract_character_regions_improved`)
   - Horizontal projection analysis
   - Character boundary detection
   - Bounding box extraction

3. **Character Recognition** (multi-method approach)
   - **CNN Model** (primary): 95% accuracy on training data
   - **Tesseract** (secondary): 61.5% accuracy with custom `gran` model
   - **Hash-based** (fallback): 28.5% accuracy using perceptual hashes

## Critical Implementation Details

### OCR Service Recognition Logic

The OCR service (`backend/application/services/ocr_service.py`) implements a sophisticated multi-method approach:

```python
# 1. CNN Model (highest accuracy)
recognized_char, cnn_confidence = self.process_with_cnn(char_image)

# 2. Tesseract (good for specific characters)
tesseract_char = self.process_with_tesseract(char_image)

# 3. Selection logic based on confidence and character type
if cnn_confidence >= 0.8:
    use CNN result
elif cnn_confidence >= 0.5:
    if tesseract_char in ['L', 'P', 'R'] and cnn_confidence < 0.7:
        use Tesseract
else:
    use Tesseract or Hash-based fallback
```

### Model Paths and Loading

The CNN model path is resolved relative to the project root:
```python
current_file = Path(__file__)
project_root = current_file.parent.parent.parent.parent
model_path = project_root / 'models' / 'cnn_model_best.pth'
```

### Preprocessing Considerations

- Training data used purple bubble backgrounds with white characters
- Test images have black backgrounds with white characters
- The preprocessing pipeline handles both cases by detecting background color
- Character thickness normalization is applied to match training data style

## Current Performance

### Recognition Accuracy (test_data/test.png - "PLEASURE")
- **Initial**: 12.5% (1/8 characters correct)
- **Current**: 62.5% (5/8 characters correct)
- **Processing time**: ~200ms

### Method-specific Accuracy
- **CNN**: 95% on training data, variable on real images
- **Tesseract**: 61.5% (16/26 characters recognized correctly)
- **Hash-based**: 28.5%

### Known Misrecognitions
- A → P (shape similarity)
- R → P (Tesseract training limitation)
- E → Z (medium CNN confidence)

## Environment Setup

### Backend Requirements
- Python 3.11+
- Tesseract OCR installed (`brew install tesseract` on macOS)
- Custom language data at `/opt/homebrew/share/tessdata/gran.traineddata`

### Frontend Requirements
- Node.js 18+
- pnpm package manager

### Optional ML Dependencies
```bash
uv pip install torch torchvision scikit-learn matplotlib tqdm seaborn
```

## Testing and Debugging

### Test Data Locations
- Real test image: `test_data/test.png` (PLEASURE text)
- Training data: `training_data/augmented/` (A-Z characters)
- Debug outputs: `tests/debug/output/`

### Common Debugging Commands
```bash
# Check model structure
python tests/debug/check_model_structure.py

# Test individual recognition methods
python tests/debug/debug_all_methods.py

# Analyze character extraction
python tests/debug/debug_char_order.py
```

## Deployment

### Frontend (Cloudflare Workers)
```bash
cd front
pnpm run build
pnpm run deploy
```

### Backend (Production)
```bash
uv pip install -e .
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

## Future Improvements

1. **Retrain CNN with real Granulate text images** (not just isolated characters)
2. **Implement ensemble voting** between recognition methods
3. **Add context-aware post-processing** for common words
4. **Optimize for real-time video processing** (current: single image focus)