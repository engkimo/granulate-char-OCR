import pytest
from backend.application.services.character_validator import CharacterValidator
from backend.domain.entities.character import Character


class TestCharacterValidator:
    def test_validate_character_valid(self):
        validator = CharacterValidator()

        char = Character("ᐁ", "A", 0.95)
        result = validator.validate_character(char)

        assert result.is_valid is True
        assert result.error is None

    def test_validate_character_invalid_mapping(self):
        validator = CharacterValidator()

        # Wrong mapping - ᐁ should map to A, not B
        char = Character("ᐁ", "B", 0.95)
        result = validator.validate_character(char)

        assert result.is_valid is False
        assert "does not match expected" in result.error

    def test_validate_character_unknown_granulate(self):
        validator = CharacterValidator()

        char = Character("X", "A", 0.95)
        result = validator.validate_character(char)

        assert result.is_valid is False
        assert "Unknown granulate symbol" in result.error

    def test_validate_character_low_confidence(self):
        validator = CharacterValidator()

        char = Character("ᐁ", "A", 0.45)
        result = validator.validate_character(char, min_confidence=0.5)

        assert result.is_valid is False
        assert "Confidence too low" in result.error

    def test_validate_character_batch(self):
        validator = CharacterValidator()

        characters = [
            Character("ᐁ", "A", 0.95),  # Valid
            Character("ᐂ", "B", 0.92),  # Valid
            Character("ᐃ", "X", 0.88),  # Invalid mapping
            Character("X", "D", 0.90),  # Unknown symbol
        ]

        results = validator.validate_batch(characters)

        assert len(results) == 4
        assert results[0].is_valid is True
        assert results[1].is_valid is True
        assert results[2].is_valid is False
        assert results[3].is_valid is False

    def test_get_correction_suggestions(self):
        validator = CharacterValidator()

        # Test correction for wrong mapping
        char = Character("ᐁ", "B", 0.95)
        suggestion = validator.get_correction_suggestion(char)

        assert suggestion is not None
        assert suggestion.latin_equivalent == "A"
        assert suggestion.confidence_adjustment == 1.0  # High confidence in correction

    def test_get_correction_suggestions_no_suggestion(self):
        validator = CharacterValidator()

        # Unknown symbol - no suggestion possible
        char = Character("X", "A", 0.95)
        suggestion = validator.get_correction_suggestion(char)

        assert suggestion is None
