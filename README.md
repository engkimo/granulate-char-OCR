# Granulate Character OCR

OCR system for recognizing fictional Granulate characters from Kamen Rider Gavv and converting them to Latin alphabet (A-Z) and numbers (0-9).

## Features

- Real-time character recognition from camera feed
- Custom Tesseract training for Granulate characters
- Clean Architecture with Python backend and modern frontend
- TDD approach for reliable code

## Quick Start

### Backend

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup and run
uv venv
uv pip install -e ".[dev]"
uv run uvicorn backend.main:app --reload
```

### Frontend

```bash
npm install
npm run dev
```

## Development

See [CLAUDE.md](CLAUDE.md) for detailed development instructions.