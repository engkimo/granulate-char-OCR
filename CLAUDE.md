# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Granulate character OCR system for recognizing fictional characters from Kamen Rider Gavv. Converts Granulate characters to Latin alphabet (A-Z) and numbers (0-9).

## Development Commands

### Backend (Python with uv)
```bash
# Setup
uv venv
uv pip install -e ".[dev]"

# Run server
uv run uvicorn backend.main:app --reload

# Testing
uv run pytest                          # Run all tests
uv run pytest backend/tests/unit/      # Run unit tests only
uv run pytest -k "test_name"           # Run specific test
uv run pytest --cov=backend            # With coverage
uv run pytest --cov=backend --cov-report=html  # HTML coverage report

# Code quality
uv run black backend/                  # Format code
uv run flake8 backend/                 # Lint
uv run mypy backend/                   # Type check
```

### Frontend (React Router + Cloudflare)
```bash
cd front

# Setup
pnpm install

# Development
pnpm dev                               # Start dev server (port 5173)

# Testing
pnpm test                              # Run tests
pnpm test:ui                           # Run with UI
pnpm test:coverage                     # Coverage report
pnpm test app/components/Camera/       # Test specific directory

# Build & Deploy
pnpm build                             # Build for production
pnpm deploy                            # Deploy to Cloudflare Pages
pnpm typecheck                         # Type checking
```

### OCR Training Data
```bash
# Extract characters from reference chart
python training_data/scripts/extract_from_reference.py

# Generate augmented data
python training_data/scripts/augment_with_gan.py

# Train few-shot learning model
python training_data/scripts/few_shot_learning.py

# Generate Tesseract training data
python training_data/scripts/create_tesseract_files.py
bash training_data/scripts/train_tesseract.sh
```

## Architecture Overview

### Backend Structure (Clean Architecture)
```
backend/
├── api/                    # FastAPI endpoints & HTTP layer
│   └── endpoints/         
├── domain/                 # Core business logic (no dependencies)
│   ├── entities/          # Character, OCRResult
│   ├── repositories/      # Abstract interfaces
│   └── use_cases/         # Business rules
├── application/            # Application services
│   ├── services/          # OCR service, validators
│   └── dto/               # Data Transfer Objects
└── infrastructure/         # External dependencies
    ├── ocr/               # Tesseract integration (placeholder)
    ├── image_processing/  # OpenCV operations
    └── mapping/           # GranulateAlphabet implementation
```

Key architectural decisions:
- Clean Architecture ensures testability - domain logic has no external dependencies
- Repository pattern allows easy mocking in tests
- Services orchestrate between layers

### Frontend Architecture
- **React Router v7** with file-based routing
- **State Management**: Zustand for local state, React Query for server state
- **API Communication**: Native fetch API (not axios - important for Cloudflare Workers compatibility)
- **Components**: Camera (WebRTC), OCRResult display
- **Deployment**: Cloudflare Pages with Workers

### API Endpoints
- `GET /api/v1/health` - Health check
- `GET /docs` - Interactive API documentation
- `POST /api/v1/ocr/process` - Process image file (multipart/form-data)
- `POST /api/v1/ocr/process-base64` - Process base64 image (JSON)

Expected OCR response format:
```json
{
  "image_id": "string",
  "text": "string",
  "average_confidence": 0.0-1.0,
  "processing_time": 0.0,
  "timestamp": "ISO 8601",
  "characters": [{
    "granulate_symbol": "string",
    "latin_equivalent": "string", 
    "confidence": 0.0-1.0
  }]
}
```

## Character Mapping System

The `GranulateAlphabet` class (singleton) manages bidirectional mappings between Granulate symbols and Latin characters. Currently uses Canadian Aboriginal Syllabics as placeholders:
- A-Z: ᐁ through ᐿ  
- 0-9: ᐀, ᑐ, ᑑ, ᑒ, ᑓ, ᑔ, ᑕ, ᑖ, ᑗ, ᐉ

## Testing Strategy

### Backend
- Domain entities have full unit test coverage
- API endpoints tested with mocked services
- Character validation includes edge cases
- Target: >90% coverage

### Frontend  
- Components tested with React Testing Library
- Browser APIs mocked in `app/test/setup.ts`
- API services tested with mocked fetch
- Vitest for fast test execution

## Current Implementation Status

**Completed:**
- ✅ Backend API structure with FastAPI
- ✅ Domain entities and character mapping
- ✅ Frontend camera integration
- ✅ OCR result display UI
- ✅ API client services
- ✅ Test infrastructure (96% backend, full frontend coverage)
- ✅ Character extraction from reference chart
- ✅ Data augmentation scripts (GAN/traditional methods)
- ✅ Few-shot learning implementation
- ✅ Apple Silicon compatibility guide

**In Progress:**
- 🔄 Tesseract OCR integration with custom model
- 🔄 Training data generation from single reference image

**TODO:**
- ⏳ WebSocket for real-time updates
- ⏳ Production deployment of trained model
- ⏳ Mobile app optimization

## Environment Variables

Frontend requires `.env` file:
```
VITE_API_URL=http://localhost:8000
```

## Development Notes

- Python 3.11+ required
- Frontend uses pnpm (not npm)
- CORS configured for development (update for production)
- All code formatted with Black (Python) and Prettier (JS/TS)
- Frontend must use fetch API instead of axios for Cloudflare Workers compatibility

### Apple Silicon Support
- PyTorch uses MPS (Metal Performance Shaders) instead of CUDA
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` for compatibility
- Optimal batch sizes: M1 (8-16), M1 Pro/Max (16-32), M2/M3 (32-64)

### Training Data Location
- Reference chart: `static/granulte_chars.jpg` (note the typo in filename)
- Extracted characters: `training_data/extracted/`
- Augmented data: `training_data/augmented/`
- Trained models: `models/`

## Machine Learning Approach

Due to limited training data (single reference chart), the project uses:
1. **Automated character extraction** from the purple bubble reference chart
2. **Advanced data augmentation**:
   - Traditional methods (rotation, scaling, noise)
   - GAN-based generation (StyleGAN2 architecture)
   - Diffusion model simulation
3. **Few-shot learning techniques**:
   - Prototypical Networks for N-way K-shot learning
   - Siamese Networks for similarity learning
   - MAML for rapid task adaptation

Expected performance:
- Initial model: 70-80% accuracy
- With few-shot learning: 85-90% accuracy
- Full pipeline: 90-95% accuracy