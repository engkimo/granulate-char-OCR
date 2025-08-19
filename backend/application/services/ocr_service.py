import uuid
import time
from typing import List
from backend.domain.entities.ocr_result import OCRResult
from backend.domain.entities.character import Character


class OCRService:
    def __init__(self):
        # TODO: Initialize Tesseract and other dependencies
        pass

    def process_image(self, image_bytes: bytes) -> OCRResult:
        # TODO: Implement actual OCR processing
        # This is a placeholder implementation

        start_time = time.time()
        image_id = f"img_{uuid.uuid4().hex[:8]}"

        # Placeholder: return empty result for now
        characters: List[Character] = []

        processing_time = time.time() - start_time

        return OCRResult(
            image_id=image_id, characters=characters, processing_time=processing_time
        )
