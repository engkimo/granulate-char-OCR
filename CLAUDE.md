# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Granulate character OCR system for recognizing fictional characters from Kamen Rider Gavv. Converts Granulate characters to Latin alphabet (A-Z) and numbers (0-9).

## Development Commands

### Backend (Python with uv)
```bash
# Initial setup
curl -LsSf https://astral.sh/uv/install.sh | sh  # Install uv if needed
uv venv
uv pip install -e ".[dev]"

# Run server
uv run uvicorn backend.main:app --reload         # Default port 8000

# Testing
uv run pytest                                    # Run all tests
uv run pytest backend/tests/unit/                # Unit tests only
uv run pytest -k "test_character_validation"     # Run specific test by name
uv run pytest --cov=backend                      # With coverage
uv run pytest --cov=backend --cov-report=html    # HTML coverage report
uv run pytest -m unit                            # Run tests by marker

# Code quality
uv run black backend/                            # Format code
uv run flake8 backend/                           # Lint
uv run mypy backend/                             # Type check
```

### Frontend (React Router + Cloudflare)
```bash
cd front

# Initial setup
pnpm install                                     # Install dependencies

# Development
pnpm dev                                         # Start dev server (port 5173)

# Testing
pnpm test                                        # Run all tests
pnpm test:ui                                     # Run with UI
pnpm test:coverage                               # Coverage report
pnpm test app/components/Camera/Camera.test.tsx  # Test specific file

# Build & Deploy
pnpm build                                       # Build for production
pnpm preview                                     # Preview production build
pnpm deploy                                      # Deploy to Cloudflare Pages
pnpm typecheck                                   # Type checking
pnpm cf-typegen                                  # Generate Cloudflare types
```

### OCR Model Training
```bash
# Prerequisites
brew install tesseract                          # macOS
sudo apt-get install tesseract-ocr pytesseract  # Ubuntu

# Character extraction from reference chart
uv run python training_data/scripts/extract_from_reference.py

# Data augmentation (creates ~150 images per character)
uv run python training_data/scripts/augment_with_gan.py

# Few-shot learning model training
uv run python training_data/scripts/few_shot_learning.py

# Tesseract training (if implementing)
python training_data/scripts/create_tesseract_files.py
bash training_data/scripts/train_tesseract.sh
```

## Architecture Overview

### Backend Structure (Clean Architecture)
```
backend/
├── api/                    # FastAPI endpoints & HTTP layer
│   └── endpoints/          # Health check, OCR endpoints
├── domain/                 # Core business logic (no external dependencies)
│   ├── entities/          # Character, OCRResult - pure data classes
│   ├── repositories/      # Abstract interfaces for data access
│   └── use_cases/         # ProcessOCRUseCase - business rules
├── application/           # Application services layer
│   ├── services/          # OCRService, CharacterValidator
│   └── dto/              # ProcessImageRequest/Response DTOs
└── infrastructure/        # External dependencies
    ├── ocr/              # TesseractOCREngine (placeholder)
    ├── image_processing/ # ImageProcessor with OpenCV
    └── mapping/          # GranulateAlphabet singleton
```

Key architectural patterns:
- **Clean Architecture**: Dependencies point inward (infrastructure → application → domain)
- **Repository Pattern**: Abstract interfaces in domain, concrete implementations in infrastructure
- **Use Case Pattern**: Each business operation is a separate use case class
- **DTO Pattern**: Separate data structures for API communication

### Frontend Architecture
- **Framework**: React Router v7 with file-based routing
- **State Management**: 
  - Zustand for local UI state (camera settings, history)
  - React Query for server state (OCR results caching)
- **API Client**: Native fetch API (required for Cloudflare Workers compatibility)
- **Component Structure**:
  - `Camera`: WebRTC integration with real-time capture
  - `OCRResult`: Display component with confidence visualization
  - `ImageUpload`: File-based alternative to camera
- **Deployment**: Cloudflare Pages with Workers for edge computing

### Training Pipeline Architecture
1. **Character Extraction** (`extract_from_reference.py`):
   - Detects purple bubbles in reference chart
   - Uses OCR to read yellow alphabet labels
   - Extracts white Granulate characters
   - Outputs 64x64 normalized images

2. **Data Augmentation** (`augment_with_gan.py`):
   - Traditional augmentation: rotation, scaling, noise, morphology
   - GAN-based generation (StyleGAN2 architecture)
   - Diffusion-style augmentation
   - Generates ~150 variations per character

3. **Few-shot Learning** (`few_shot_learning.py`):
   - Prototypical Networks for N-way K-shot learning
   - Siamese Networks for similarity learning
   - MAML for rapid adaptation
   - Combined model architecture

## API Contract

### OCR Processing Endpoint
```
POST /api/v1/ocr/process
Content-Type: multipart/form-data
Body: image (file)

Response:
{
  "image_id": "uuid-v4",
  "text": "HELLO",
  "average_confidence": 0.92,
  "processing_time": 0.256,
  "timestamp": "2024-01-01T12:00:00Z",
  "characters": [
    {
      "granulate_symbol": "ᐈ",
      "latin_equivalent": "H",
      "confidence": 0.95
    }
  ]
}
```

## Character Mapping System

The `GranulateAlphabet` singleton manages bidirectional mappings:
- Uses Canadian Aboriginal Syllabics as placeholder symbols
- A-Z mapped to ᐁ through ᐿ
- 0-9 mapped to ᐀, ᑐ, ᑑ, ᑒ, ᑓ, ᑔ, ᑕ, ᑖ, ᑗ, ᐉ
- Thread-safe singleton pattern implementation

## Critical Implementation Details

### Backend
- OCR processing currently returns placeholder results (not yet integrated with Tesseract)
- Image preprocessing uses OpenCV for grayscale conversion and enhancement
- All domain entities use Pydantic for validation
- FastAPI auto-generates OpenAPI docs at `/docs`

### Frontend
- Must use fetch API, not axios (Cloudflare Workers requirement)
- Camera access requires HTTPS in production
- WebRTC MediaStream API used for real-time capture
- React Router's new v7 data loading patterns

### Training Scripts
- Requires Tesseract installed system-wide for OCR
- Character extraction depends on detecting yellow text in purple bubbles
- Apple Silicon users need `PYTORCH_ENABLE_MPS_FALLBACK=1` for PyTorch
- Training data excluded from git (see .gitignore)

## Testing Approach

### Backend Testing
- Unit tests: Domain logic with no dependencies
- Integration tests: API endpoints with mocked services  
- Use pytest markers: `@pytest.mark.unit`, `@pytest.mark.integration`
- Current coverage: 96%

### Frontend Testing
- Component tests with React Testing Library
- API service tests with mocked fetch
- Browser APIs (MediaDevices, Canvas) mocked in test setup
- Vitest for fast execution

## Environment Configuration

### Backend
No `.env` file needed for development (uses defaults)

### Frontend (.env)
```
VITE_API_URL=http://localhost:8000
```

## Known Issues & Limitations

1. OCR processing returns empty results (Tesseract integration pending)
2. Reference chart image has typo: `granulte_chars.jpg` (missing 'a')
3. Numbers (0-9) not included in reference chart
4. WebSocket support not implemented
5. No authentication/authorization system

## Apple Silicon Compatibility

For M1/M2/M3 Macs:
- PyTorch uses MPS instead of CUDA
- Set `export PYTORCH_ENABLE_MPS_FALLBACK=1`
- Optimal batch sizes:
  - M1: 8-16
  - M1 Pro/Max: 16-32  
  - M2/M3: 32-64
- Use `torch.device("mps")` instead of `torch.device("cuda")`