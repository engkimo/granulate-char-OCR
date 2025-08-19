# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Granulate character OCR system for recognizing fictional characters from Kamen Rider Gavv. The system converts Granulate characters to their corresponding Latin alphabet (A-Z) and numbers (0-9) equivalents, similar to Google Translate's camera feature.

## Development Commands

### Initial Setup
```bash
npm install              # Install dependencies
npm run dev             # Start development server with hot reload
npm run build           # Build for production
npm run preview         # Preview production build locally
npm run lint            # Run ESLint
npm run typecheck       # Run TypeScript type checking
npm test               # Run all tests
npm test:watch         # Run tests in watch mode
```

### OCR Training Data
```bash
npm run train:prepare   # Prepare training data for Tesseract
npm run train:generate  # Generate .traineddata file
npm run train:validate  # Validate trained model accuracy
```

## Architecture

### Core Components

1. **OCR Engine** (`src/core/ocr/`)
   - `TesseractWrapper.ts` - Tesseract.js integration with custom traineddata
   - `ImagePreprocessor.ts` - OpenCV.js-based image enhancement
   - `CharacterDetector.ts` - EAST text detection for character regions

2. **Character Mapping** (`src/core/mapping/`)
   - `GranulateAlphabet.ts` - Character mapping dictionary (A-Z, 0-9)
   - `CharacterValidator.ts` - Validates detected characters against known set

3. **Camera Integration** (`src/features/camera/`)
   - Uses MediaStream API with WebRTC for real-time processing
   - Processes frames at 30fps using Web Workers

4. **State Management**
   - Zustand for React or Pinia for Vue (depending on framework choice)
   - Stores recognition history, settings, and UI state

### Key Technical Decisions

- **Client-side only**: All processing happens in the browser (no server required)
- **Web Workers**: OCR processing runs in background threads to maintain UI responsiveness
- **Progressive Web App**: Works offline once loaded
- **Custom traineddata**: Granulate-specific Tesseract training data (~10MB compressed)

## Project-Specific Notes

1. **Character Set**: 36 total characters (A-Z + 0-9), no lowercase distinction
2. **Recognition Target**: Printed Granulate text only (not handwritten)
3. **Performance Goal**: <1 second recognition time, 85%+ accuracy
4. **Browser Constraints**: iOS Safari camera API limitations require special handling

## Testing Approach

- Unit tests for character mapping logic
- Integration tests for OCR pipeline
- E2E tests for camera → recognition flow
- Visual regression tests for UI components
- Performance benchmarks for recognition speed

## Deployment

The application is deployed as a static site to Vercel/Netlify with GitHub Actions CI/CD pipeline. Production builds are automatically deployed on tagged releases.