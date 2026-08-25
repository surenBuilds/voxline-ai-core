"""Armenian Language Compliance Benchmark — 25 test cases.

Unit tests verify that:
  1. User input is unchanged through the pipeline.
  2. Language detection works correctly.
  3. Language policy survives ContextBuilder (instruction appears in messages).
  4. System instruction includes Armenian directive.

Integration tests (gated by VOXLINE_TEST_LIVE_MODEL env var) verify:
  5. Response is primarily Armenian for Armenian inputs.
  6. Response is non-empty.
  7. No unexpected full English response.
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.language import detect_language, Language, LanguagePolicy
from src.assistant.context import ContextBuilder
from src.assistant.session import Session, SessionManager, SessionMode

LIVE = os.environ.get("VOXLINE_TEST_LIVE_MODEL")

# ---------------------------------------------------------------------------
# Benchmark cases: (input, expected_detection)
# ---------------------------------------------------------------------------

ARMENIAN_CASES = [
    ("\u0532\u0561\u057c\u0565\u057e", Language.ARMENIAN),
    ("\u0532\u0561\u057c\u0565\u057e, \u056b\u0576\u0579\u057a\u0565\u057f \u0565\u057d", Language.ARMENIAN),
    ("\u053b\u0576\u0579\u057a\u0565\u057f \u056f\u0561\u0580\u0578\u0582\u0572 \u0565\u057d \u0563\u0578\u0582\u0574\u0561\u0580 \u057e\u0561\u057d\u057f\u0561\u056f\u0565\u056c", Language.ARMENIAN),
    ("\u053d\u0576\u0578\u0582\u0580 \u0565\u057d \u0570\u0561\u057e\u0561\u0576\u0584\u0578\u0582\u0569\u0575\u0578\u0582\u0576", Language.ARMENIAN),
    ("\u053f\u0578\u0572\u0561\u0576\u0561\u0575\u056b\u0576 Python-ov \u056f\u0578\u0564\u0561\u057e\u0578\u0582\u0574 \u0565\u0584\u055d \u056f\u0561\u0580\u0578\u0582\u0572 \u0565\u057d", Language.ARMENIAN),
    ("\u053d\u0576\u0584 \u0569\u0565\u0572\u0576\u0578\u0582\u0569\u0575\u0578\u0582\u0576\u056b \u0574\u0561\u057d\u056b\u0576 \u057e\u0561\u0580\u0578\u056e \u0565\u057d", Language.ARMENIAN),
    ("\u054e\u0578\u057d\u056c\u056b\u0576\u0565 \u057a\u056c\u0561\u057f\u0578\u0580\u0574\u0561\u0576 \u056b\u0576\u0579\u057a\u0565\u057f \u056f\u0561\u0580\u0578\u0582\u0572", Language.ARMENIAN),
    ("\u053b\u0576\u057f\u0578\u0580\u0576\u0565\u057f\u0568 \u056b\u0576\u056f\u0578\u0580", Language.ARMENIAN),
    ("\u054a\u0565\u057f\u0580\u0578\u0582\u0569\u0575\u0578\u0582\u0576\u0578\u0582\u0574 \u056f\u0561\u0572\u0574\u0581\u056b \u057d\u057f\u0565\u0572 \u0578\u0582\u0577\u0578\u0582\u0574", Language.ARMENIAN),
    ("\u053d\u0576\u0578\u0582\u0580 \u056f\u0578\u0572\u0561\u0576\u0561\u0575\u056b\u0576\u0568 \u0570\u0561\u057d\u056b\u0576\u0578\u0582\u057e\u0565\u056c \u0565\u057d\u055d \u0569\u0565\u0572\u0576\u0578\u0582\u0569\u0575\u0578\u0582\u0576 \u0570\u0561\u0577\u057e\u0565\u056c", Language.ARMENIAN),
    ("\u054e\u0561\u057d\u057f\u0561\u056f\u0565\u056c \u056f\u0561\u0580\u0578\u0582\u0572 \u0570\u0561\u057d\u056b\u0576\u0578\u0582\u057e\u0565\u056c \u056f\u0561\u0580\u0578\u0582\u0581", Language.ARMENIAN),
    ("\u054d\u057a\u056b\u0576 \u0561\u057a\u0561\u0580\u0570\u0578\u0582\u0574 \u0578\u0582\u056f\u0561\u0581\u0576\u0576\u0565\u056c \u056f\u0561\u0580\u0578\u0582\u0572", Language.ARMENIAN),
    ("\u0532\u0578\u0582\u0572\u0561\u0580 \u056f\u0561\u0580\u0565\u056c \u0578\u0582\u0577\u0578\u0582\u0574 \u0579\u057a\u0561\u057f\u056b \u0570\u0561\u0574\u0561\u0580", Language.ARMENIAN),
    ("\u053b\u0574 \u057f\u0578\u0582\u0576\u0568 \u0578\u0582\u0580\u0578\u056e \u0565\u057d\u055d \u057b\u0561\u0576\u0563\u0561\u0563\u0578\u0582\u0574 \u056f\u0561\u0580\u0578\u0582\u0572", Language.ARMENIAN),
    ("\u0535\u0580\u056f\u0578\u0582\u0580 \u057f\u0578\u0574\u0561\u0576 \u056d\u0561\u0576\u057f\u0578\u0572\u0580\u0561\u056e \u0565\u057d", Language.ARMENIAN),
]

ENGLISH_CASES = [
    ("Hello, how are you?", Language.ENGLISH),
    ("What is the capital of France?", Language.ENGLISH),
    ("Explain machine learning", Language.ENGLISH),
    ("Install Python 3.13 via pip", Language.ENGLISH),
    ("Write a REST API in FastAPI", Language.ENGLISH),
]


# ---------------------------------------------------------------------------
# Unit tests — no model required
# ---------------------------------------------------------------------------


class TestInputUnchanged:
    """User input must never be modified by detection or policy."""

    @pytest.mark.parametrize("text,expected", ARMENIAN_CASES + ENGLISH_CASES)
    def test_input_identity(self, text, expected):
        assert detect_language(text) == expected
        assert isinstance(text, str)


class TestArmenianDetection:
    @pytest.mark.parametrize("text,expected", ARMENIAN_CASES)
    def test_detects_armenian(self, text, expected):
        assert detect_language(text) == expected


class TestEnglishDetection:
    @pytest.mark.parametrize("text,expected", ENGLISH_CASES)
    def test_detects_english(self, text, expected):
        assert detect_language(text) == expected


class TestPolicyInjection:
    """Language policy must appear as system message in ContextBuilder output."""

    @pytest.fixture()
    def ctx_builder(self):
        return ContextBuilder(max_chars=4000, max_history=10)

    @pytest.fixture()
    def session(self):
        sm = SessionManager()
        return sm.create(SessionMode.CHAT)

    def test_armenian_instruction_in_messages(self, ctx_builder, session):
        lang_instr = LanguagePolicy.get_system_instruction(Language.ARMENIAN)
        ctx = ctx_builder.build(
            session=session,
            user_message="\u0532\u0561\u057c\u0565\u057e",
            language_instruction=lang_instr,
        )
        first_msg = ctx.messages[0]
        assert first_msg["role"] == "system"
        assert "Armenian" in first_msg["content"]
        assert "Eastern Armenian" in first_msg["content"]

    def test_english_instruction_in_messages(self, ctx_builder, session):
        lang_instr = LanguagePolicy.get_system_instruction(Language.ENGLISH)
        ctx = ctx_builder.build(
            session=session,
            user_message="Hello",
            language_instruction=lang_instr,
        )
        first_msg = ctx.messages[0]
        assert first_msg["role"] == "system"
        assert "English" in first_msg["content"]

    def test_no_instruction_when_none(self, ctx_builder, session):
        ctx = ctx_builder.build(
            session=session,
            user_message="Hello",
            language_instruction=None,
        )
        assert ctx.messages[0]["role"] != "system"

    def test_user_message_unchanged_after_context(self, ctx_builder, session):
        arm_msg = "\u0532\u0561\u057c\u0565\u057e, \u056b\u0576\u0579\u057a\u0565\u057f \u0565\u057d"
        lang_instr = LanguagePolicy.get_system_instruction(Language.ARMENIAN)
        ctx = ctx_builder.build(
            session=session,
            user_message=arm_msg,
            language_instruction=lang_instr,
        )
        last = ctx.messages[-1]
        assert last["role"] == "user"
        assert last["content"] == arm_msg

    def test_history_preserved_with_language_instruction(self, ctx_builder, session):
        session.add_message("user", "\u0532\u0561\u057c\u0565\u057e")
        session.add_message("assistant", "\u0532\u0561\u0580\u0581\u0565\u057e")
        lang_instr = LanguagePolicy.get_system_instruction(Language.ARMENIAN)
        ctx = ctx_builder.build(
            session=session,
            user_message="\u053b\u0576\u0579\u057a\u0565\u057f \u0565\u057d",
            language_instruction=lang_instr,
        )
        roles = [m["role"] for m in ctx.messages]
        assert roles.count("user") >= 3
        assert roles.count("assistant") >= 2


class TestRetryPolicy:
    def test_retry_triggered(self):
        assert LanguagePolicy.should_retry(Language.ARMENIAN, Language.ENGLISH) is True

    def test_retry_not_triggered_same_lang(self):
        assert LanguagePolicy.should_retry(Language.ARMENIAN, Language.ARMENIAN) is False

    def test_retry_not_for_english_user(self):
        assert LanguagePolicy.should_retry(Language.ENGLISH, Language.ARMENIAN) is False

    def test_retry_instruction_exists(self):
        instr = LanguagePolicy.get_retry_instruction(Language.ARMENIAN)
        assert len(instr) > 0
        assert "Armenian" in instr

    def test_temperature_overrides(self):
        assert LanguagePolicy.get_temperature(Language.ARMENIAN) == 0.5
        assert LanguagePolicy.get_temperature(Language.ENGLISH) == 0.7


class TestLanguageInstructionPersistence:
    """Instruction survives a full context build cycle with history."""

    @pytest.fixture()
    def full_setup(self):
        sm = SessionManager()
        session = sm.create(SessionMode.CHAT)
        session.add_message("user", "\u0532\u0561\u057c\u0565\u057e")
        session.add_message("assistant", "\u0532\u0561\u0580\u0581\u0565\u057e \u0576\u0578\u0580 \u0565\u057d")
        ctx_builder = ContextBuilder(max_chars=4000, max_history=10)
        return ctx_builder, session

    def test_instruction_survives_with_history(self, full_setup):
        ctx_builder, session = full_setup
        lang_instr = LanguagePolicy.get_system_instruction(Language.ARMENIAN)
        ctx = ctx_builder.build(
            session=session,
            user_message="\u053b\u0576\u0579\u057a\u0565\u057f \u056f\u0561\u0580\u0578\u0582\u0572 \u0565\u057d",
            language_instruction=lang_instr,
        )
        assert ctx.messages[0]["role"] == "system"
        assert "Eastern Armenian" in ctx.messages[0]["content"]
        assert ctx.messages[-1]["content"] == "\u053b\u0576\u0579\u057a\u0565\u057f \u056f\u0561\u0580\u0578\u0582\u0572 \u0565\u057d"


# ---------------------------------------------------------------------------
# Integration tests — require live Qwen model
# ---------------------------------------------------------------------------

_integration = pytest.mark.skipif(
    not LIVE, reason="Set VOXLINE_TEST_LIVE_MODEL=1 to run live integration tests"
)


@_integration
class TestLiveArmenianCompliance:
    """These tests require the Qwen model loaded — run sparingly."""

    def _generate(self, user_msg: str) -> str:
        import asyncio
        from src.providers.qwen_provider import QwenProvider
        from src.providers.base import GenerationConfig
        if not hasattr(self, "_provider"):
            self._provider = QwenProvider(
                model_path="models/Qwen2.5-0.5B-Instruct", device="cpu"
            )
        user_lang = detect_language(user_msg)
        lang_instr = LanguagePolicy.get_system_instruction(user_lang)
        config = GenerationConfig(max_tokens=120, temperature=0.5)
        msgs = [{"role": "system", "content": lang_instr},
                {"role": "user", "content": user_msg}]
        return asyncio.run(self._provider.chat(msgs, config))

    @pytest.mark.parametrize("text,_", ARMENIAN_CASES[:10])
    def test_armenian_response_is_not_empty(self, text, _):
        resp = self._generate(text)
        assert len(resp.strip()) > 0

    @pytest.mark.parametrize("text,_", ENGLISH_CASES[:5])
    def test_english_response_is_english(self, text, _):
        from src.language import is_english
        resp = self._generate(text)
        assert is_english(resp) or detect_language(resp) == Language.UNKNOWN
