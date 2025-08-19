import pytest
from backend.infrastructure.mapping.granulate_alphabet import GranulateAlphabet


class TestGranulateAlphabet:
    def test_singleton_pattern(self):
        alphabet1 = GranulateAlphabet()
        alphabet2 = GranulateAlphabet()
        assert alphabet1 is alphabet2

    def test_get_latin_equivalent_valid(self):
        alphabet = GranulateAlphabet()

        # Test some known mappings
        assert alphabet.get_latin_equivalent("ᐁ") == "A"
        assert alphabet.get_latin_equivalent("ᐂ") == "B"
        assert alphabet.get_latin_equivalent("ᐿ") == "Z"
        assert alphabet.get_latin_equivalent("᐀") == "0"
        assert alphabet.get_latin_equivalent("ᑐ") == "1"
        assert alphabet.get_latin_equivalent("ᐉ") == "9"

    def test_get_latin_equivalent_invalid(self):
        alphabet = GranulateAlphabet()

        assert alphabet.get_latin_equivalent("X") is None
        assert alphabet.get_latin_equivalent("") is None
        assert alphabet.get_latin_equivalent("ABC") is None

    def test_get_granulate_symbol_valid(self):
        alphabet = GranulateAlphabet()

        assert alphabet.get_granulate_symbol("A") == "ᐁ"
        assert alphabet.get_granulate_symbol("B") == "ᐂ"
        assert alphabet.get_granulate_symbol("Z") == "ᐿ"
        assert alphabet.get_granulate_symbol("0") == "᐀"
        assert alphabet.get_granulate_symbol("1") == "ᑐ"
        assert alphabet.get_granulate_symbol("9") == "ᐉ"

    def test_get_granulate_symbol_invalid(self):
        alphabet = GranulateAlphabet()

        assert alphabet.get_granulate_symbol("a") is None  # lowercase
        assert alphabet.get_granulate_symbol("") is None
        assert alphabet.get_granulate_symbol("AB") is None
        assert alphabet.get_granulate_symbol("@") is None

    def test_is_valid_granulate_symbol(self):
        alphabet = GranulateAlphabet()

        assert alphabet.is_valid_granulate_symbol("ᐁ") is True
        assert alphabet.is_valid_granulate_symbol("ᐿ") is True
        assert alphabet.is_valid_granulate_symbol("᐀") is True

        assert alphabet.is_valid_granulate_symbol("X") is False
        assert alphabet.is_valid_granulate_symbol("") is False
        assert alphabet.is_valid_granulate_symbol("ᐁᐂ") is False

    def test_is_valid_latin_character(self):
        alphabet = GranulateAlphabet()

        assert alphabet.is_valid_latin_character("A") is True
        assert alphabet.is_valid_latin_character("Z") is True
        assert alphabet.is_valid_latin_character("0") is True
        assert alphabet.is_valid_latin_character("9") is True

        assert alphabet.is_valid_latin_character("a") is False
        assert alphabet.is_valid_latin_character("") is False
        assert alphabet.is_valid_latin_character("AB") is False
        assert alphabet.is_valid_latin_character("@") is False

    def test_all_mappings_count(self):
        alphabet = GranulateAlphabet()
        # 26 letters + 10 digits = 36 total
        assert len(alphabet.get_all_mappings()) == 36

    def test_bidirectional_mapping_consistency(self):
        alphabet = GranulateAlphabet()
        mappings = alphabet.get_all_mappings()

        for granulate, latin in mappings.items():
            assert alphabet.get_granulate_symbol(latin) == granulate
            assert alphabet.get_latin_equivalent(granulate) == latin
