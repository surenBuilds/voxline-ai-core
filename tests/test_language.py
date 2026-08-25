"""Tests for src/language.py — detection + policy."""
from src.language import detect_language, is_armenian, is_english, Language, LanguagePolicy


class TestDetectArmenian:
    def test_basic_greeting(self):
        assert detect_language("\u0532\u0561\u057c\u0565\u057e") == Language.ARMENIAN

    def test_question(self):
        assert detect_language("\u053b\u0576\u0579\u057a\u0565\u057f \u0565\u057d") == Language.ARMENIAN

    def test_long_sentence(self):
        assert detect_language(
            "\u053b\u0576\u0579\u057a\u0565\u057f \u056f\u0561\u0580\u0578\u0582\u0572 "
            "\u0565\u057d \u0563\u0578\u0582\u0574\u0561\u0580 \u057e\u0561\u057d\u057f\u0561\u056f\u0565\u056c"
        ) == Language.ARMENIAN

    def test_with_punctuation(self):
        assert detect_language("\u0532\u0561\u057c\u0565\u057e\u0554 \u056b\u0576\u0579\u057a\u0565\u057f \u0565\u057d\u0561\u0576") == Language.ARMENIAN

    def test_with_numbers(self):
        assert detect_language("\u0532\u0561\u057c\u0565\u057e 5 \u0570\u0561\u057e\u056b\u0581") == Language.ARMENIAN

    def test_mixed_armenian_english_dominant_armenian(self):
        text = "\u053b\u0576\u0579\u057a\u0565\u057f \u056f\u0561\u0580\u0578\u0582\u0572 API-\u056b \u0574\u056b\u057d\u057f"
        assert detect_language(text) == Language.ARMENIAN

    def test_technical_armenian(self):
        text = "\u053f\u0578\u0572\u0561\u0576\u0561\u0575\u056b\u0576 Python \u057a\u0580\u0563\u0580\u0561\u057e\u0578\u0582\u0574"
        assert detect_language(text) == Language.ARMENIAN


class TestDetectEnglish:
    def test_basic(self):
        assert detect_language("Hello, how are you?") == Language.ENGLISH

    def test_long(self):
        assert detect_language("The quick brown fox jumps over the lazy dog") == Language.ENGLISH

    def test_technical(self):
        assert detect_language("Install Python 3.13 via pip") == Language.ENGLISH


class TestDetectUnknown:
    def test_empty(self):
        assert detect_language("") == Language.UNKNOWN

    def test_whitespace(self):
        assert detect_language("   ") == Language.UNKNOWN

    def test_numbers_only(self):
        assert detect_language("12345") == Language.UNKNOWN

    def test_cyrillic_only(self):
        assert detect_language("\u041f\u0440\u0438\u0432\u0435\u0442") == Language.UNKNOWN

    def test_none_input(self):
        assert detect_language(None) == Language.UNKNOWN


class TestIsArmenian:
    def test_true(self):
        assert is_armenian("\u0532\u0561\u057c\u0565\u057e") is True

    def test_false_english(self):
        assert is_armenian("Hello") is False

    def test_false_unknown(self):
        assert is_armenian("") is False


class TestIsEnglish:
    def test_true(self):
        assert is_english("Hello") is True

    def test_false_armenian(self):
        assert is_english("\u0532\u0561\u057c\u0565\u057e") is False


class TestLanguagePolicy:
    def test_armenian_instruction(self):
        instr = LanguagePolicy.get_system_instruction(Language.ARMENIAN)
        assert "Armenian" in instr
        assert "Eastern Armenian" in instr

    def test_english_instruction(self):
        instr = LanguagePolicy.get_system_instruction(Language.ENGLISH)
        assert "English" in instr

    def test_retry_instruction(self):
        instr = LanguagePolicy.get_retry_instruction(Language.ARMENIAN)
        assert "Armenian" in instr
        assert len(instr) > 0

    def test_temperature_armenian(self):
        temp = LanguagePolicy.get_temperature(Language.ARMENIAN)
        assert temp == 0.5

    def test_temperature_english(self):
        temp = LanguagePolicy.get_temperature(Language.ENGLISH)
        assert temp == 0.7

    def test_should_retry_armenian_user_english_response(self):
        assert LanguagePolicy.should_retry(Language.ARMENIAN, Language.ENGLISH) is True

    def test_should_not_retry_armenian_user_armenian_response(self):
        assert LanguagePolicy.should_retry(Language.ARMENIAN, Language.ARMENIAN) is False

    def test_should_not_retry_english_user(self):
        assert LanguagePolicy.should_retry(Language.ENGLISH, Language.ARMENIAN) is False

    def test_should_not_retry_unknown_user(self):
        assert LanguagePolicy.should_retry(Language.UNKNOWN, Language.ENGLISH) is False


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
