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

# Development
pnpm install
pnpm dev                               # Start dev server

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
- **API Communication**: Axios with typed services
- **Components**: Camera (WebRTC), OCRResult display
- **Deployment**: Cloudflare Pages with Workers

### API Endpoints
- `GET /api/v1/health` - Health check
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
- API services tested with mocked axios
- Vitest for fast test execution

## Current Implementation Status

**Completed:**
- ✅ Backend API structure with FastAPI
- ✅ Domain entities and character mapping
- ✅ Frontend camera integration
- ✅ OCR result display UI
- ✅ API client services
- ✅ Test infrastructure (96% backend, full frontend coverage)

**Placeholder/TODO:**
- ⏳ Actual Tesseract OCR processing (returns empty results)
- ⏳ Image preprocessing pipeline
- ⏳ Granulate character training data
- ⏳ WebSocket for real-time updates

## Environment Variables

Frontend requires:
```
VITE_API_URL=http://localhost:8000
```

## Development Notes

- Python 3.11+ required
- Frontend uses pnpm (not npm)
- CORS configured for development (update for production)
- All code formatted with Black (Python) and Prettier (JS/TS)