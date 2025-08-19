from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import base64
from typing import List, Optional
from backend.application.services.ocr_service import OCRService


router = APIRouter()


class Base64ImageRequest(BaseModel):
    image: str


class CharacterResponse(BaseModel):
    granulate_symbol: str
    latin_equivalent: str
    confidence: float


class OCRResponse(BaseModel):
    image_id: str
    text: str
    average_confidence: float
    processing_time: float
    characters: List[CharacterResponse]


@router.post("/process", response_model=OCRResponse)
async def process_image(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Only images are allowed.",
        )

    try:
        # Read file content
        content = await file.read()

        # Process with OCR service
        ocr_service = OCRService()
        result = ocr_service.process_image(content)

        # Convert to response model
        return OCRResponse(
            image_id=result.image_id,
            text=result.text,
            average_confidence=result.average_confidence,
            processing_time=result.processing_time,
            characters=[
                CharacterResponse(
                    granulate_symbol=char.granulate_symbol,
                    latin_equivalent=char.latin_equivalent,
                    confidence=char.confidence,
                )
                for char in result.characters
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@router.post("/process-base64", response_model=OCRResponse)
async def process_image_base64(request: Base64ImageRequest):
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(request.image)

        # Process with OCR service
        ocr_service = OCRService()
        result = ocr_service.process_image(image_bytes)

        # Convert to response model
        return OCRResponse(
            image_id=result.image_id,
            text=result.text,
            average_confidence=result.average_confidence,
            processing_time=result.processing_time,
            characters=[
                CharacterResponse(
                    granulate_symbol=char.granulate_symbol,
                    latin_equivalent=char.latin_equivalent,
                    confidence=char.confidence,
                )
                for char in result.characters
            ],
        )
    except base64.binascii.Error:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
