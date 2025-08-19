from typing import Dict, Optional


class GranulateAlphabet:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_mappings()
        return cls._instance

    def _initialize_mappings(self):
        # Granulate to Latin mappings
        # Using Canadian Aboriginal Syllabics as placeholder for actual Granulate characters
        self._granulate_to_latin: Dict[str, str] = {
            # Letters A-Z
            "ᐁ": "A",
            "ᐂ": "B",
            "ᐃ": "C",
            "ᐄ": "D",
            "ᐅ": "E",
            "ᐆ": "F",
            "ᐇ": "G",
            "ᐈ": "H",
            "ᐊ": "I",
            "ᐋ": "J",
            "ᐌ": "K",
            "ᐍ": "L",
            "ᐎ": "M",
            "ᐏ": "N",
            "ᐐ": "O",
            "ᐑ": "P",
            "ᐒ": "Q",
            "ᐓ": "R",
            "ᐔ": "S",
            "ᐕ": "T",
            "ᐖ": "U",
            "ᐗ": "V",
            "ᐘ": "W",
            "ᐙ": "X",
            "ᐚ": "Y",
            "ᐿ": "Z",
            # Numbers 0-9 (using different symbols)
            "᐀": "0",
            "ᑐ": "1",
            "ᑑ": "2",
            "ᑒ": "3",
            "ᑓ": "4",
            "ᑔ": "5",
            "ᑕ": "6",
            "ᑖ": "7",
            "ᑗ": "8",
            "ᐉ": "9",
        }

        # Create reverse mapping
        self._latin_to_granulate: Dict[str, str] = {
            v: k for k, v in self._granulate_to_latin.items()
        }

    def get_latin_equivalent(self, granulate_symbol: str) -> Optional[str]:
        if not granulate_symbol or len(granulate_symbol) != 1:
            return None
        return self._granulate_to_latin.get(granulate_symbol)

    def get_granulate_symbol(self, latin_char: str) -> Optional[str]:
        if not latin_char or len(latin_char) != 1:
            return None
        return self._latin_to_granulate.get(latin_char)

    def is_valid_granulate_symbol(self, symbol: str) -> bool:
        return symbol in self._granulate_to_latin

    def is_valid_latin_character(self, char: str) -> bool:
        return char in self._latin_to_granulate

    def get_all_mappings(self) -> Dict[str, str]:
        return self._granulate_to_latin.copy()
