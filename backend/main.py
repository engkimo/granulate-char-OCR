from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.endpoints import ocr, health


app = FastAPI(
    title="Granulate Character OCR API",
    description="OCR system for recognizing Granulate characters from Kamen Rider Gavv",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(ocr.router, prefix="/api/v1/ocr", tags=["ocr"])


@app.get("/")
async def root():
    return {
        "message": "Granulate Character OCR API",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
