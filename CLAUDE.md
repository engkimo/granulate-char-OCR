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
./train_tesseract.sh                            # Run automated training script
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
- **CNN model**: 100% validation accuracy, ~95% test accuracy
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
3. **Character Recognition** (in order of accuracy):
   - Primary: CNN model (`models/cnn_model_best.pth`)
   - Secondary: Few-shot learning model (`models/prototypical_network.pth`)
   - Fallback: Hash-based matching using `granulate_alphabet_generated.py`
   - Future: Custom Tesseract model (requires training completion)

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

1. **Apple Silicon PyTorch**: MPS backend may have issues with certain operations. Use CPU fallback:
   ```python
   device = 'cpu'  # Instead of 'mps' if errors occur
   ```

2. **iOS Safari Camera**: Requires specific WebRTC constraints:
   ```javascript
   { video: { facingMode: 'environment', width: { ideal: 1920 } } }
   ```

3. **Tesseract Language Data**: Custom `.traineddata` file must be in tessdata directory

4. **Training Data**: Reference image filename typo: `granulte_chars.jpg` (missing 'a')

5. **Model Integration**: CNN and few-shot models not yet integrated into OCR API

6. **Character Extraction**: Manual sorting was required due to grid detection issues

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
npm run build
npx wrangler deploy  # Uses wrangler.jsonc configuration
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