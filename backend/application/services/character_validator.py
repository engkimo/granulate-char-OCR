from dataclasses import dataclass
from typing import List, Optional
from backend.domain.entities.character import Character
from backend.infrastructure.mapping.granulate_alphabet import GranulateAlphabet


@dataclass
class ValidationResult:
    is_valid: bool
    error: Optional[str] = None


@dataclass
class CorrectionSuggestion:
    latin_equivalent: str
    confidence_adjustment: float


class CharacterValidator:
    def __init__(self):
        self._alphabet = GranulateAlphabet()

    def validate_character(
        self, character: Character, min_confidence: float = 0.5
    ) -> ValidationResult:
        # Check if granulate symbol is known
        if not self._alphabet.is_valid_granulate_symbol(character.granulate_symbol):
            return ValidationResult(
                is_valid=False,
                error=f"Unknown granulate symbol: {character.granulate_symbol}",
            )

        # Check confidence threshold
        if character.confidence < min_confidence:
            return ValidationResult(
                is_valid=False,
                error=f"Confidence too low: {character.confidence:.2f} < {min_confidence}",
            )

        # Check if mapping is correct
        expected_latin = self._alphabet.get_latin_equivalent(character.granulate_symbol)
        if expected_latin != character.latin_equivalent:
            return ValidationResult(
                is_valid=False,
                error=f"Granulate symbol {character.granulate_symbol} does not match expected mapping. Expected: {expected_latin}, Got: {character.latin_equivalent}",
            )

        return ValidationResult(is_valid=True)

    def validate_batch(
        self, characters: List[Character], min_confidence: float = 0.5
    ) -> List[ValidationResult]:
        return [self.validate_character(char, min_confidence) for char in characters]

    def get_correction_suggestion(
        self, character: Character
    ) -> Optional[CorrectionSuggestion]:
        # Can only suggest correction if we know the granulate symbol
        if not self._alphabet.is_valid_granulate_symbol(character.granulate_symbol):
            return None

        correct_latin = self._alphabet.get_latin_equivalent(character.granulate_symbol)

        # If already correct, no suggestion needed
        if correct_latin == character.latin_equivalent:
            return None

        return CorrectionSuggestion(
            latin_equivalent=correct_latin,
            confidence_adjustment=1.0,  # High confidence in dictionary-based correction
        )
