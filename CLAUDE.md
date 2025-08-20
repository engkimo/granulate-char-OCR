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
# Initial setup
npm install                                       # Install dependencies

# Development
npm run dev                                      # Start dev server with hot reload
npm run build                                    # Build for production
npm run preview                                  # Preview production build locally

# Testing
npm test                                         # Run all tests
npm test:watch                                   # Run tests in watch mode
npm test -- Camera                               # Run specific test file

# Code quality
npm run lint                                     # Run ESLint
npm run typecheck                               # Run TypeScript type checking

# Deployment (Cloudflare Workers)
npm run deploy                                   # Deploy to Cloudflare Workers
npm run build-cf-types                          # Generate Cloudflare types
```

### OCR Model Training

```bash
# Character extraction and data preparation
python training_data/scripts/extract_from_reference.py    # Extract characters from reference image
python training_data/scripts/augment_with_gan.py         # Generate augmented training data
python training_data/scripts/few_shot_learning.py        # Train few-shot learning models
python training_data/scripts/create_mapping.py           # Create character mappings

# Tesseract training (requires Tesseract installed)
cd training_data
tesseract [input.tif] [output] -l eng --psm 10 makebox  # Generate box files
# ... (see docs/training-guide.md for full process)
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
├── extracted/          # Characters extracted from reference image
├── augmented/         # Data augmentation output (~150 variations/char)
└── scripts/
    ├── extract_from_reference.py  # HoughCircles + OCR extraction
    ├── augment_with_gan.py       # Traditional + GAN augmentation
    └── few_shot_learning.py      # Prototypical/Siamese networks
```

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
1. **Image Preprocessing**: Gaussian blur → Adaptive threshold → Morphological operations
2. **Text Detection**: EAST model or contour detection for character regions
3. **Character Recognition**: 
   - Primary: Custom Tesseract model trained on Granulate characters
   - Fallback: Hash-based matching using `granulate_alphabet_generated.py`
   - Enhancement: Few-shot learning models for difficult cases

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

4. **Training Data**: Early見表 image has filename typo: `granulte_chars.jpg` (missing 'a')

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
- **Frontend**: Component behavior tests over implementation details
- **OCR Models**: Accuracy benchmarks on test dataset
- **E2E**: Camera → OCR → Display flow testing

When implementing new features, maintain the Clean Architecture boundaries and ensure proper separation of concerns.