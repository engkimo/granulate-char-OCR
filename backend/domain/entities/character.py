from dataclasses import dataclass
import re


@dataclass
class Character:
    granulate_symbol: str
    latin_equivalent: str
    confidence: float

    def __post_init__(self):
        self._validate()

    def _validate(self):
        if not re.match(r"^[A-Z0-9]$", self.latin_equivalent):
            raise ValueError("Latin equivalent must be a single alphanumeric character")

        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")

    def __str__(self):
        return f"Character({self.granulate_symbol} → {self.latin_equivalent}, confidence: {self.confidence * 100:.1f}%)"

    def __eq__(self, other):
        if not isinstance(other, Character):
            return False
        return (
            self.granulate_symbol == other.granulate_symbol
            and self.latin_equivalent == other.latin_equivalent
            and self.confidence == other.confidence
        )
