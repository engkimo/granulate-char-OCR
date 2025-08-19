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

**Placeholder/TODO:**
- ⏳ Actual Tesseract OCR processing (returns empty results)
- ⏳ Image preprocessing pipeline
- ⏳ Granulate character training data
- ⏳ WebSocket for real-time updates

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

## Development History

### Phase 1: Backend Development (Completed)
1. **Architecture Design**
   - Implemented Clean Architecture with TDD approach
   - Created domain entities (Character, OCRResult)
   - Built infrastructure layer with GranulateAlphabet mapping
   - Developed application services (CharacterValidator)

2. **API Implementation**
   - Set up FastAPI with CORS middleware
   - Created REST endpoints for OCR processing
   - Implemented health check endpoint
   - Added interactive API docs at /docs

3. **Testing**
   - Achieved 96% test coverage
   - 32 backend tests passing
   - Used pytest with coverage reporting

### Phase 2: Frontend Development (Completed)
1. **Initial Setup**
   - Created React Router app with Cloudflare Pages template
   - Configured Vitest for testing
   - Set up Tailwind CSS

2. **Core Components**
   - Camera component with WebRTC integration
   - OCR result display with confidence indicators
   - Navigation between main, history, and settings pages

3. **State Management**
   - Zustand for local state (OCR results, history)
   - React Query for server state management

4. **API Integration**
   - Initially used axios but encountered Cloudflare Workers compatibility issues
   - Migrated to native fetch API to resolve SSR errors
   - All API calls now use fetch with proper error handling

5. **Testing**
   - 21 frontend tests passing
   - Mocked browser APIs (MediaDevices, Canvas)
   - Component tests with React Testing Library

### Key Technical Decisions
1. **Python Environment**: Used `uv` for faster dependency management
2. **Frontend Framework**: React Router v7 for Cloudflare Pages compatibility
3. **API Client**: Switched from axios to fetch API due to Node.js import errors in Cloudflare Workers
4. **Character Mapping**: Used Canadian Aboriginal Syllabics as placeholder Granulate characters

### Current Issues Resolved
- ✅ Fixed missing route files (history.tsx, settings.tsx)
- ✅ Resolved Cloudflare Workers Node.js compatibility errors by removing axios
- ✅ Fixed test mocking for Canvas and MediaDevices APIs

### Known Limitations
- OCR processing returns empty results (placeholder implementation)
- No actual Tesseract integration yet
- Character training data not generated
- WebSocket support not implemented