from dataclasses import dataclass, field
from datetime import datetime
from typing import List
from backend.domain.entities.character import Character


@dataclass
class OCRResult:
    image_id: str
    characters: List[Character]
    processing_time: float
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def text(self) -> str:
        return "".join(char.latin_equivalent for char in self.characters)

    @property
    def average_confidence(self) -> float:
        if not self.characters:
            return 0.0
        return sum(char.confidence for char in self.characters) / len(self.characters)

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "text": self.text,
            "average_confidence": self.average_confidence,
            "processing_time": self.processing_time,
            "timestamp": self.timestamp.isoformat(),
            "characters": [
                {
                    "granulate_symbol": char.granulate_symbol,
                    "latin_equivalent": char.latin_equivalent,
                    "confidence": char.confidence,
                }
                for char in self.characters
            ],
        }
