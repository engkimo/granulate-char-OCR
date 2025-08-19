import pytest
from datetime import datetime
from backend.domain.entities.ocr_result import OCRResult
from backend.domain.entities.character import Character


class TestOCRResult:
    def test_create_ocr_result(self):
        characters = [
            Character("ᐁ", "A", 0.95),
            Character("ᐂ", "B", 0.92),
            Character("ᐃ", "C", 0.88),
        ]

        result = OCRResult(
            image_id="img_123", characters=characters, processing_time=0.256
        )

        assert result.image_id == "img_123"
        assert len(result.characters) == 3
        assert result.processing_time == 0.256
        assert isinstance(result.timestamp, datetime)

    def test_ocr_result_text_property(self):
        characters = [
            Character("ᐁ", "A", 0.95),
            Character("ᐂ", "B", 0.92),
            Character("ᐃ", "C", 0.88),
        ]

        result = OCRResult("img_123", characters, 0.256)
        assert result.text == "ABC"

    def test_ocr_result_average_confidence(self):
        characters = [
            Character("ᐁ", "A", 0.90),
            Character("ᐂ", "B", 0.80),
            Character("ᐃ", "C", 0.70),
        ]

        result = OCRResult("img_123", characters, 0.256)
        assert abs(result.average_confidence - 0.80) < 0.0001

    def test_ocr_result_empty_characters(self):
        result = OCRResult("img_123", [], 0.256)
        assert result.text == ""
        assert result.average_confidence == 0.0

    def test_ocr_result_to_dict(self):
        characters = [Character("ᐁ", "A", 0.95), Character("ᐂ", "B", 0.92)]

        result = OCRResult("img_123", characters, 0.256)
        result_dict = result.to_dict()

        assert result_dict["image_id"] == "img_123"
        assert result_dict["text"] == "AB"
        assert abs(result_dict["average_confidence"] - 0.935) < 0.0001
        assert result_dict["processing_time"] == 0.256
        assert len(result_dict["characters"]) == 2
        assert "timestamp" in result_dict
