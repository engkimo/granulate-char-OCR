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

### OCR Model Training & Testing

```bash
# Character extraction and data preparation (完了済み)
python training_data/scripts/extract_from_reference.py    # Extract characters from reference image
python training_data/scripts/augment_simple.py           # Generate augmented training data (150 variations/char)
python training_data/scripts/create_mapping.py           # Create character mappings

# Machine Learning Models (PyTorch required)
uv pip install torch torchvision scikit-learn matplotlib tqdm seaborn
python training_data/scripts/few_shot_learning.py        # Train few-shot learning models
python training_data/scripts/train_cnn_model.py          # Train CNN model for higher accuracy

# Image preprocessing optimization
python training_data/scripts/optimize_preprocessing.py   # Evaluate and optimize preprocessing methods

# Testing recognition accuracy
python test_hash_mapping.py                      # Test hash-based recognition (19.2% accuracy)
python test_similarity_mapping.py                # Test similarity-based recognition (28.5% accuracy)
python test_ocr_api.py                          # Test OCR API with current models

# Tesseract training (requires Tesseract installed)
python training_data/scripts/create_tesseract_data.py    # Generate training data for Tesseract
cd training_data/tesseract
python convert_to_white_bg.py                    # Convert images to white background (required)
./robust_train.sh                                # Run robust training for all 26 characters
python test_all_chars.py                         # Test recognition accuracy (61.5% achieved)
```

## Architecture

### Backend Structure (Clean Architecture)

```
backend/
├── domain/              # Business logic, no external dependencies
│   ├── entities/        # Core business objects (Image, OCRResult)
│   └── repositories/    # Interface definitions
├── application/         # Use cases and business rules
│   ├── services/        # OCRService with business logic
│   └── use_cases/       # ProcessImageUseCase
├── infrastructure/      # External dependencies implementation
│   ├── ocr/            # Tesseract/ML model integration
│   ├── mapping/        # Granulate character mappings
│   └── repositories/   # Concrete implementations
└── api/                # FastAPI routes and DTOs
    ├── routers/        # OCR endpoints
    └── models/         # Request/Response models
```

**Key Patterns:**
- Dependency Injection: Services depend on abstract repositories
- Use Case pattern: Each operation is a distinct use case
- Repository pattern: Abstract data access behind interfaces
- DTO pattern: Separate API models from domain entities

### Frontend Structure (React Router v7)

```
front/
├── app/
│   ├── routes/         # Route components
│   ├── components/     # Reusable UI components
│   │   └── Camera/     # Enhanced camera with image processing
│   ├── features/       # Feature-specific components
│   ├── services/       # API client and external services
│   └── stores/         # Zustand state management
└── public/            # Static assets
```

**Key Technologies:**
- React Router v7 with Cloudflare Workers deployment
- Zustand for state management
- TanStack Query for server state
- WebRTC/MediaStream API for camera access
- Canvas API for image processing

**Camera Component Features:**
- Auto-flip for front-facing camera
- 4 processing modes: none, basic, enhanced, aggressive (default)
- Advanced camera constraints (ISO, focus, white balance)
- Real-time preview with processing applied
- Captured image history (last 5 images)
- Aggressive mode: Binary threshold at 128 (same as zutomayo_OCR)

### Training Data Pipeline

```
training_data/
├── extracted/          # Characters extracted from reference image (26 chars, A-Z)
├── augmented/         # Data augmentation output (~150 variations/char, 3,926 total)
├── tesseract/         # Tesseract training data (TIFF, BOX, unicharset)
└── scripts/
    ├── extract_from_reference.py    # Grid-based extraction from purple bubbles
    ├── augment_simple.py           # PyTorch-free augmentation (rotation, scale, noise)
    ├── create_mapping.py           # Generate perceptual hash mappings
    ├── few_shot_learning.py        # Prototypical Networks (5-way 5-shot)
    ├── train_cnn_model.py          # Custom CNN architecture (100% val accuracy)
    ├── optimize_preprocessing.py    # Evaluate 7 preprocessing methods
    └── create_tesseract_data.py   # Generate Tesseract training files
```

### Model Performance

- **Hash-based recognition**: 19.2% accuracy (8x8 perceptual hash)
- **Similarity-based recognition**: 28.5% accuracy (Hamming distance threshold)
- **Tesseract model**: 61.5% accuracy (16/26 characters recognized)
  - Successfully recognized: A, B, C, E, G, I, L, N, O, P, Q, U, V, X, Y, Z
  - Failed recognition: D, F, H, J, K, M, R, S, T, W
- **CNN model**: 100% validation accuracy, ~95% test accuracy (NOT YET INTEGRATED)
- **Preprocessing**: Best method is "contrast_enhance" (score: 0.204)

## API Contracts

### OCR Endpoint
```
POST /api/ocr/process
Content-Type: multipart/form-data

Request:
  - file: Image file (JPEG/PNG)
  - options: { enhance: boolean, language: "granulate" }

Response:
  {
    "text": "HELLO",
    "confidence": 0.95,
    "processing_time": 0.234,
    "character_details": [
      { "char": "H", "confidence": 0.98, "position": {...} }
    ]
  }
```

### WebSocket Real-time OCR
```
WS /api/ocr/stream
Message: { "type": "frame", "data": "base64_image" }
Response: { "type": "result", "text": "ABC", "confidence": 0.89 }
```

## Critical Implementation Details

### Character Recognition Flow
1. **Image Preprocessing**: Optimized pipeline using contrast enhancement
   - Bilateral filter for noise reduction
   - CLAHE for contrast enhancement  
   - Adaptive thresholding
   - Morphological operations for cleanup
2. **Text Detection**: Contour detection for character regions
3. **Character Recognition** (in order of priority):
   - Primary: Tesseract with custom `gran` language model (61.5% accuracy) - INTEGRATED
   - Fallback: Hash-based matching using `granulate_alphabet_generated.py` (28.5% accuracy) - INTEGRATED
   - Future Primary: CNN model (`models/cnn_model_best.pth`, 95% accuracy) - NOT YET INTEGRATED
   - Future Secondary: Few-shot learning model (`models/prototypical_network.pth`) - NOT YET INTEGRATED

**Current Status**: OCRService implements Tesseract + Hash-based fallback. CNN integration recommended for better accuracy.

### Granulate Alphabet Mapping
- 36 characters total: A-Z (26) + 0-9 (10)
- Hash-based mapping in `backend/infrastructure/mapping/granulate_alphabet_generated.py`
- Each character has a 64-bit perceptual hash for robust matching
- No lowercase distinction in Granulate script

### Performance Targets
- Recognition latency: <1 second per image
- Accuracy target: 85%+ on clear images
- Real-time video: 15+ FPS processing

## Known Issues & Limitations

1. **OCR Service Partially Implemented**: The backend OCRService (`backend/application/services/ocr_service.py`) now uses:
   - Tesseract with custom `gran` language (61.5% accuracy)
   - Hash-based mapping as fallback (28.5% accuracy)
   - CNN model integration still needed for better accuracy (95%)

2. **Apple Silicon PyTorch**: MPS backend may have issues with certain operations. Use CPU fallback:
   ```python
   device = 'cpu'  # Instead of 'mps' if errors occur
   ```

3. **iOS Safari Camera**: Requires specific WebRTC constraints:
   ```javascript
   { video: { facingMode: 'environment', width: { ideal: 1920 } } }
   ```

4. **Tesseract Language Data**: Custom `gran.traineddata` installed at `/opt/homebrew/share/tessdata/`

5. **Training Data**: Reference image filename typo: `granulte_chars.jpg` (missing 'a')

6. **Model Integration**: CNN and few-shot models not yet integrated into OCR API

7. **Character Extraction**: Manual sorting was required due to grid detection issues

8. **Chrome DevTools Error**: Harmless error about `/.well-known/appspecific/com.chrome.devtools.json` can be ignored

## Environment Variables

### Frontend (.env)
```
VITE_API_URL=http://localhost:8000  # Backend API URL
```

### Backend
- No required environment variables (uses defaults)
- Optional: `TESSDATA_PREFIX` for custom Tesseract data location

## Deployment

### Frontend (Cloudflare Workers)
```bash
cd front
pnpm run build
pnpm run deploy  # Uses wrangler.jsonc configuration
```

### Backend (Production)
```bash
uv pip install -e .  # Production dependencies only
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

## Testing Philosophy

- **Backend**: Domain logic heavily unit tested, integration tests for API endpoints
- **Frontend**: Component behavior tests using Vitest and React Testing Library
- **OCR Models**: Accuracy benchmarks on test dataset
- **E2E**: Camera → OCR → Display flow testing

## Project Dependencies

### Python (Backend)
- **Python 3.11+** required
- **uv** package manager for fast dependency management
- **FastAPI** for REST API
- **OpenCV** and **Pillow** for image processing
- **PyTorch** for machine learning models
- **pytesseract** for OCR integration

### Node.js (Frontend)
- **pnpm** package manager
- **React 19** with React Router v7
- **Vite** for build tooling
- **Cloudflare Workers** for deployment
- **Zustand** for state management
- **TanStack Query** for server state

When implementing new features, maintain the Clean Architecture boundaries and ensure proper separation of concerns.

## Improving OCR Accuracy

The OCR service is implemented with Tesseract (61.5% accuracy) and Hash-based fallback (28.5% accuracy).

### Current Implementation in `backend/application/services/ocr_service.py`:
```python
# Primary: Tesseract with custom model
recognized_char = self.process_with_tesseract(char_image)

# Fallback: Hash-based recognition
if not recognized_char:
    recognized_char = self.alphabet.compare_image_to_mapping(char_image)
```

### To Achieve 95% Accuracy - Integrate CNN Model:
```python
import torch
model = torch.load('models/cnn_model_best.pth')
# Add CNN recognition as primary method before Tesseract
```

### Tesseract Training Notes:
- Images must have white background (use `convert_to_white_bg.py`)
- Training data location: `training_data/tesseract/`
- Model location: `/opt/homebrew/share/tessdata/gran.traineddata`