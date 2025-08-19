from fastapi import APIRouter
from datetime import datetime


router = APIRouter()


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "0.1.0",
        "service": "granulate-char-ocr",
    }
