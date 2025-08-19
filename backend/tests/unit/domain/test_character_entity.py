import pytest
from backend.domain.entities.character import Character


class TestCharacter:
    def test_create_character_with_valid_data(self):
        character = Character(
            granulate_symbol="ᐁ", latin_equivalent="A", confidence=0.95
        )

        assert character.granulate_symbol == "ᐁ"
        assert character.latin_equivalent == "A"
        assert character.confidence == 0.95

    def test_character_validation_latin_equivalent(self):
        with pytest.raises(
            ValueError, match="Latin equivalent must be a single alphanumeric character"
        ):
            Character(granulate_symbol="ᐁ", latin_equivalent="AB", confidence=0.95)

    def test_character_validation_confidence_range(self):
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            Character(granulate_symbol="ᐁ", latin_equivalent="A", confidence=1.5)

    def test_character_equality(self):
        char1 = Character("ᐁ", "A", 0.95)
        char2 = Character("ᐁ", "A", 0.95)
        char3 = Character("ᐂ", "B", 0.90)

        assert char1 == char2
        assert char1 != char3

    def test_character_string_representation(self):
        character = Character("ᐁ", "A", 0.95)
        assert str(character) == "Character(ᐁ → A, confidence: 95.0%)"
