"""
Tests for Phase 7 Step 5-6: BusinessAssistant.
"""

import os
import tempfile
import unittest

from src.assistant.business import (
    ActionItem,
    BusinessAssistant,
    BusinessContext,
    BusinessPlan,
    BusinessRequest,
    BusinessResponse,
    BusinessTaskType,
    KPI,
    Priority,
    Recommendation,
)
from src.assistant.context import ContextBuilder
from src.assistant.session import Session, SessionManager, SessionMode
from src.errors import SessionNotFoundError, ProviderError
from src.memory.memory import MemoryStore
from src.providers.base import (
    AIProvider,
    GenerationConfig,
    ModelInfo,
    ProviderHealth,
    ProviderStatus,
)


# ---------------------------------------------------------------------------
# FakeProvider (no model loading)
# ---------------------------------------------------------------------------


class FakeProvider(AIProvider):
    def __init__(self, response: str = "Business analysis complete."):
        self._response = response
        self.call_count = 0
        self.last_messages = None
        self.last_config = None

    @property
    def provider_id(self) -> str:
        return "fake"

    @property
    def model_id(self) -> str:
        return "fake-model-v1"

    @property
    def supports_streaming(self) -> bool:
        return False

    async def generate(self, prompt: str, config: GenerationConfig) -> str:
        self.call_count += 1
        return self._response

    async def chat(
        self, messages, config: GenerationConfig
    ) -> str:
        self.call_count += 1
        self.last_messages = messages
        self.last_config = config
        return self._response

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(
            status=ProviderStatus.HEALTHY,
            message="Fake provider is healthy",
        )

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            model_id=self.model_id,
            provider_id=self.provider_id,
            model_type="fake",
        )


class FailingProvider(AIProvider):
    @property
    def provider_id(self):
        return "fail"

    @property
    def model_id(self):
        return "fail-m"

    @property
    def supports_streaming(self):
        return False

    async def generate(self, prompt, config):
        raise RuntimeError("model crashed")

    async def chat(self, messages, config):
        raise RuntimeError("model crashed")

    async def health_check(self):
        return ProviderHealth(
            status=ProviderStatus.UNAVAILABLE, message="down"
        )

    def get_model_info(self):
        return ModelInfo(
            model_id="fail-m", provider_id="fail", model_type="fail"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_assistant(
    response: str = "Business analysis.", memory_store=None
):
    provider = FakeProvider(response=response)
    sm = SessionManager()
    ctx = ContextBuilder(memory_store=memory_store)
    return BusinessAssistant(
        provider=provider,
        session_manager=sm,
        context_builder=ctx,
        memory_store=memory_store,
    )


def _session_for(asst: BusinessAssistant) -> Session:
    return asst.new_session()


def _tmp_store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return MemoryStore(db_path=path), path


# ---------------------------------------------------------------------------
# 1. Initialization
# ---------------------------------------------------------------------------


class TestBusinessAssistantInit(unittest.TestCase):

    def test_initialization(self):
        asst = _make_assistant()
        self.assertIsNotNone(asst.provider)
        self.assertIsNotNone(asst.session_manager)

    def test_initialization_with_memory(self):
        store, _ = _tmp_store()
        asst = _make_assistant(memory_store=store)
        self.assertIs(asst.memory_store, store)
        store.close()


# ---------------------------------------------------------------------------
# 2-8. Task type analysis
# ---------------------------------------------------------------------------


class TestBusinessTaskTypes(unittest.TestCase):

    def test_general_analysis(self):
        asst = _make_assistant(response="General analysis done.")
        session = _session_for(asst)
        req = BusinessRequest(
            task_type=BusinessTaskType.GENERAL_ANALYSIS,
            user_request="Analyze our current situation.",
        )
        resp = asst.analyze(session.session_id, req)
        self.assertIsInstance(resp, BusinessResponse)
        self.assertEqual(resp.task_type, BusinessTaskType.GENERAL_ANALYSIS)
        self.assertEqual(resp.response, "General analysis done.")

    def test_company_analysis(self):
        asst = _make_assistant()
        session = _session_for(asst)
        req = BusinessRequest(
            task_type=BusinessTaskType.COMPANY_ANALYSIS,
            user_request="Analyze our company strengths and weaknesses.",
        )
        resp = asst.analyze(session.session_id, req)
        self.assertEqual(resp.task_type, BusinessTaskType.COMPANY_ANALYSIS)

    def test_sales(self):
        asst = _make_assistant()
        session = _session_for(asst)
        req = BusinessRequest(
            task_type=BusinessTaskType.SALES,
            user_request="How can we improve our sales pipeline?",
        )
        resp = asst.analyze(session.session_id, req)
        self.assertEqual(resp.task_type, BusinessTaskType.SALES)

    def test_market_analysis(self):
        asst = _make_assistant()
        session = _session_for(asst)
        req = BusinessRequest(
            task_type=BusinessTaskType.MARKET_ANALYSIS,
            user_request="What is the market opportunity for our product?",
        )
        resp = asst.analyze(session.session_id, req)
        self.assertEqual(resp.task_type, BusinessTaskType.MARKET_ANALYSIS)

    def test_strategy(self):
        asst = _make_assistant()
        session = _session_for(asst)
        req = BusinessRequest(
            task_type=BusinessTaskType.STRATEGY,
            user_request="What should our 3-year strategy be?",
        )
        resp = asst.analyze(session.session_id, req)
        self.assertEqual(resp.task_type, BusinessTaskType.STRATEGY)

    def test_kpi_analysis(self):
        asst = _make_assistant()
        session = _session_for(asst)
        req = BusinessRequest(
            task_type=BusinessTaskType.KPI_ANALYSIS,
            user_request="Analyze our KPIs.",
            business_context=BusinessContext(
                current_kpis=[
                    KPI(name="Revenue", value=100000, target=150000, unit="USD"),
                    KPI(name="Churn", value=5.2, target=3.0, unit="%"),
                ]
            ),
        )
        resp = asst.analyze(session.session_id, req)
        self.assertEqual(resp.task_type, BusinessTaskType.KPI_ANALYSIS)

    def test_action_plan(self):
        asst = _make_assistant()
        session = _session_for(asst)
        req = BusinessRequest(
            task_type=BusinessTaskType.ACTION_PLAN,
            user_request="Create a plan to reduce costs.",
        )
        resp = asst.analyze(session.session_id, req)
        self.assertEqual(resp.task_type, BusinessTaskType.ACTION_PLAN)


# ---------------------------------------------------------------------------
# 9. BusinessContext injection
# ---------------------------------------------------------------------------


class TestBusinessContextInjection(unittest.TestCase):

    def test_context_reaches_provider(self):
        provider = FakeProvider(response="ok")
        sm = SessionManager()
        ctx_builder = ContextBuilder()
        asst = BusinessAssistant(
            provider=provider,
            session_manager=sm,
            context_builder=ctx_builder,
        )
        session = asst.new_session()
        biz_ctx = BusinessContext(
            company_name="Acme Corp",
            industry="Technology",
            current_kpis=[KPI(name="Revenue", value=50000, unit="USD")],
        )
        req = BusinessRequest(
            task_type=BusinessTaskType.COMPANY_ANALYSIS,
            user_request="Analyze our company.",
            business_context=biz_ctx,
        )
        asst.analyze(session.session_id, req)
        self.assertIsNotNone(provider.last_messages)
        all_content = str(provider.last_messages)
        self.assertIn("Acme Corp", all_content)
        self.assertIn("Technology", all_content)

    def test_no_context_still_works(self):
        asst = _make_assistant()
        session = _session_for(asst)
        req = BusinessRequest(
            task_type=BusinessTaskType.GENERAL_ANALYSIS,
            user_request="General question.",
        )
        resp = asst.analyze(session.session_id, req)
        self.assertEqual(resp.response, "Business analysis.")


# ---------------------------------------------------------------------------
# 10. BusinessRequest validation
# ---------------------------------------------------------------------------


class TestBusinessRequestValidation(unittest.TestCase):

    def test_empty_request_rejected(self):
        asst = _make_assistant()
        session = _session_for(asst)
        req = BusinessRequest(
            task_type=BusinessTaskType.GENERAL_ANALYSIS,
            user_request="",
        )
        with self.assertRaises(ValueError):
            asst.analyze(session.session_id, req)

    def test_whitespace_request_rejected(self):
        asst = _make_assistant()
        session = _session_for(asst)
        req = BusinessRequest(
            task_type=BusinessTaskType.GENERAL_ANALYSIS,
            user_request="   ",
        )
        with self.assertRaises(ValueError):
            asst.analyze(session.session_id, req)

    def test_invalid_task_type(self):
        asst = _make_assistant()
        session = _session_for(asst)
        req = BusinessRequest(
            task_type="not_a_valid_type",
            user_request="test",
        )
        with self.assertRaises(ValueError):
            asst.analyze(session.session_id, req)


# ---------------------------------------------------------------------------
# 11. BusinessResponse structure
# ---------------------------------------------------------------------------


class TestBusinessResponseStructure(unittest.TestCase):

    def test_response_fields(self):
        asst = _make_assistant()
        session = _session_for(asst)
        req = BusinessRequest(
            task_type=BusinessTaskType.STRATEGY,
            user_request="Strategy question.",
            language="en",
        )
        resp = asst.analyze(session.session_id, req)
        self.assertEqual(resp.response, "Business analysis.")
        self.assertEqual(resp.task_type, BusinessTaskType.STRATEGY)
        self.assertEqual(resp.session_id, session.session_id)
        self.assertEqual(resp.provider_id, "fake")
        self.assertEqual(resp.model_id, "fake-model-v1")
        self.assertEqual(resp.language, "en")
        self.assertIsInstance(resp.metadata, dict)
        self.assertIn("history_length", resp.metadata)

    def test_response_defaults(self):
        resp = BusinessResponse(
            response="text",
            task_type=BusinessTaskType.GENERAL_ANALYSIS,
            session_id="s1",
            provider_id="p1",
            model_id="m1",
        )
        self.assertIsNone(resp.language)
        self.assertIsNone(resp.recommendations)
        self.assertIsNone(resp.action_items)
        self.assertIsNone(resp.risks)
        self.assertIsNone(resp.assumptions)
        self.assertIsNone(resp.confidence)


# ---------------------------------------------------------------------------
# 12-13. Armenian and English requests
# ---------------------------------------------------------------------------


class TestLanguageSupport(unittest.TestCase):

    def test_english_request(self):
        asst = _make_assistant(response="English response.")
        session = _session_for(asst)
        req = BusinessRequest(
            task_type=BusinessTaskType.GENERAL_ANALYSIS,
            user_request="Analyze our business.",
            language="en",
        )
        resp = asst.analyze(session.session_id, req)
        self.assertEqual(resp.response, "English response.")
        self.assertEqual(resp.language, "en")

    def test_armenian_request(self):
        asst = _make_assistant(response="Հայերեն պատասխան։")
        session = _session_for(asst)
        req = BusinessRequest(
            task_type=BusinessTaskType.GENERAL_ANALYSIS,
            user_request="Վերլուծիր մեր բիզնեսը։",
            language="hy",
        )
        resp = asst.analyze(session.session_id, req)
        self.assertEqual(resp.response, "Հայերեն պատասխան։")
        self.assertEqual(resp.language, "hy")

    def test_armenian_kpi_analysis(self):
        asst = _make_assistant(response="KPI վերլուծություն։")
        session = _session_for(asst)
        req = BusinessRequest(
            task_type=BusinessTaskType.KPI_ANALYSIS,
            user_request="Վերլուծիր KPI-ները։",
            language="hy",
            business_context=BusinessContext(
                company_name="Վոքսլայն",
                current_kpis=[
                    KPI(name="Եկամուտ", value=100000, unit="USD"),
                ],
            ),
        )
        resp = asst.analyze(session.session_id, req)
        self.assertEqual(resp.language, "hy")


# ---------------------------------------------------------------------------
# 14. Session isolation
# ---------------------------------------------------------------------------


class TestSessionIsolation(unittest.TestCase):

    def test_business_session_not_chat(self):
        sm = SessionManager()
        chat_session = sm.create(SessionMode.CHAT)
        provider = FakeProvider()
        ctx = ContextBuilder()
        asst = BusinessAssistant(
            provider=provider, session_manager=sm, context_builder=ctx
        )
        req = BusinessRequest(
            task_type=BusinessTaskType.GENERAL_ANALYSIS,
            user_request="Business question.",
        )
        with self.assertRaises(ValueError) as cm:
            asst.analyze(chat_session.session_id, req)
        self.assertIn("BUSINESS", str(cm.exception))

    def test_separate_sessions_do_not_mix(self):
        sm = SessionManager()
        provider = FakeProvider()
        ctx = ContextBuilder()
        asst = BusinessAssistant(
            provider=provider, session_manager=sm, context_builder=ctx
        )
        s1 = asst.new_session()
        s2 = asst.new_session()
        req = BusinessRequest(
            task_type=BusinessTaskType.SALES,
            user_request="Sales question for s1.",
        )
        asst.analyze(s1.session_id, req)
        req2 = BusinessRequest(
            task_type=BusinessTaskType.MARKET_ANALYSIS,
            user_request="Market question for s2.",
        )
        asst.analyze(s2.session_id, req2)
        self.assertEqual(len(s1.history), 2)
        self.assertEqual(len(s2.history), 2)
        self.assertIn("s1", s1.history[0]["content"])
        self.assertIn("s2", s2.history[0]["content"])


# ---------------------------------------------------------------------------
# 15. Memory integration
# ---------------------------------------------------------------------------


class TestMemoryIntegration(unittest.TestCase):

    def test_explicit_persist(self):
        store, _ = _tmp_store()
        asst = _make_assistant(memory_store=store)
        session = _session_for(asst)
        req = BusinessRequest(
            task_type=BusinessTaskType.STRATEGY,
            user_request="Our strategy is to focus on enterprise.",
        )
        asst.analyze(session.session_id, req, persist_memory=True)
        results = store.search_memories("enterprise")
        self.assertGreater(len(results), 0)
        store.close()

    def test_no_persist_by_default(self):
        store, _ = _tmp_store()
        asst = _make_assistant(memory_store=store)
        session = _session_for(asst)
        req = BusinessRequest(
            task_type=BusinessTaskType.GENERAL_ANALYSIS,
            user_request="Quick question.",
        )
        asst.analyze(session.session_id, req)
        results = store.search_memories("quick question")
        self.assertEqual(len(results), 0)
        store.close()

    def test_keyword_triggers_persist(self):
        store, _ = _tmp_store()
        asst = _make_assistant(memory_store=store)
        session = _session_for(asst)
        req = BusinessRequest(
            task_type=BusinessTaskType.COMPANY_ANALYSIS,
            user_request="Our company goal is to reach $1M revenue.",
        )
        asst.analyze(session.session_id, req)
        results = store.search_memories("company")
        self.assertGreater(len(results), 0)
        store.close()

    def test_business_context_triggers_persist(self):
        store, _ = _tmp_store()
        asst = _make_assistant(memory_store=store)
        session = _session_for(asst)
        req = BusinessRequest(
            task_type=BusinessTaskType.COMPANY_ANALYSIS,
            user_request="Analyze our company.",
            business_context=BusinessContext(company_name="Acme"),
        )
        asst.analyze(session.session_id, req)
        results = store.search_memories("company")
        self.assertGreater(len(results), 0)
        tags = results[0].tags
        self.assertIn("business", tags)
        self.assertIn("acme", tags)
        store.close()

    def test_no_memory_store(self):
        asst = _make_assistant(memory_store=None)
        session = _session_for(asst)
        req = BusinessRequest(
            task_type=BusinessTaskType.GENERAL_ANALYSIS,
            user_request="remember our revenue target",
        )
        resp = asst.analyze(session.session_id, req, persist_memory=True)
        self.assertEqual(resp.response, "Business analysis.")


# ---------------------------------------------------------------------------
# 16. Provider invocation
# ---------------------------------------------------------------------------


class TestProviderInvocation(unittest.TestCase):

    def test_provider_called_once(self):
        provider = FakeProvider(response="ok")
        sm = SessionManager()
        asst = BusinessAssistant(provider=provider, session_manager=sm)
        session = asst.new_session()
        req = BusinessRequest(
            task_type=BusinessTaskType.GENERAL_ANALYSIS,
            user_request="test",
        )
        asst.analyze(session.session_id, req)
        self.assertEqual(provider.call_count, 1)

    def test_provider_receives_messages(self):
        provider = FakeProvider(response="ok")
        sm = SessionManager()
        asst = BusinessAssistant(provider=provider, session_manager=sm)
        session = asst.new_session()
        req = BusinessRequest(
            task_type=BusinessTaskType.STRATEGY,
            user_request="strategy question",
        )
        asst.analyze(session.session_id, req)
        self.assertIsNotNone(provider.last_messages)
        self.assertGreater(len(provider.last_messages), 0)

    def test_provider_last_message_is_user(self):
        provider = FakeProvider(response="ok")
        sm = SessionManager()
        asst = BusinessAssistant(provider=provider, session_manager=sm)
        session = asst.new_session()
        req = BusinessRequest(
            task_type=BusinessTaskType.SALES,
            user_request="sales question",
        )
        asst.analyze(session.session_id, req)
        last = provider.last_messages[-1]
        self.assertEqual(last["role"], "user")
        self.assertEqual(last["content"], "sales question")

    def test_system_instruction_sent(self):
        provider = FakeProvider(response="ok")
        sm = SessionManager()
        asst = BusinessAssistant(provider=provider, session_manager=sm)
        session = asst.new_session()
        req = BusinessRequest(
            task_type=BusinessTaskType.GENERAL_ANALYSIS,
            user_request="test",
        )
        asst.analyze(session.session_id, req)
        all_content = str(provider.last_messages)
        self.assertIn("professional business analyst", all_content.lower())


# ---------------------------------------------------------------------------
# 17. Provider failure
# ---------------------------------------------------------------------------


class TestProviderFailure(unittest.TestCase):

    def test_provider_failure_raises(self):
        sm = SessionManager()
        asst = BusinessAssistant(
            provider=FailingProvider(), session_manager=sm
        )
        session = asst.new_session()
        req = BusinessRequest(
            task_type=BusinessTaskType.GENERAL_ANALYSIS,
            user_request="test",
        )
        with self.assertRaises(ProviderError):
            asst.analyze(session.session_id, req)


# ---------------------------------------------------------------------------
# 18. Missing business context
# ---------------------------------------------------------------------------


class TestMissingBusinessContext(unittest.TestCase):

    def test_none_business_context(self):
        asst = _make_assistant()
        session = _session_for(asst)
        req = BusinessRequest(
            task_type=BusinessTaskType.COMPANY_ANALYSIS,
            user_request="Analyze our company.",
            business_context=None,
        )
        resp = asst.analyze(session.session_id, req)
        self.assertIsNotNone(resp.response)
        self.assertFalse(resp.metadata.get("has_business_context"))

    def test_empty_business_context(self):
        asst = _make_assistant()
        session = _session_for(asst)
        req = BusinessRequest(
            task_type=BusinessTaskType.MARKET_ANALYSIS,
            user_request="Market analysis.",
            business_context=BusinessContext(),
        )
        resp = asst.analyze(session.session_id, req)
        self.assertIsNotNone(resp.response)


# ---------------------------------------------------------------------------
# 19. Missing KPI values
# ---------------------------------------------------------------------------


class TestMissingKPIValues(unittest.TestCase):

    def test_kpi_with_no_value(self):
        kpi = KPI(name="Retention", target=95.0, unit="%")
        ctx_str = kpi.to_context()
        self.assertIn("Retention", ctx_str)
        self.assertIn("NOT PROVIDED", ctx_str)
        self.assertIn("95.0", ctx_str)

    def test_kpi_with_no_target(self):
        kpi = KPI(name="Revenue", value=100000, unit="USD")
        ctx_str = kpi.to_context()
        self.assertIn("100000", ctx_str)
        self.assertIn("NOT PROVIDED", ctx_str)

    def test_kpi_full(self):
        kpi = KPI(name="MRR", value=50000, target=60000, unit="USD", period="monthly")
        ctx_str = kpi.to_context()
        self.assertIn("50000", ctx_str)
        self.assertIn("60000", ctx_str)
        self.assertIn("monthly", ctx_str)

    def test_kpi_nothing_provided(self):
        kpi = KPI(name="Unknown Metric")
        ctx_str = kpi.to_context()
        self.assertIn("NOT PROVIDED", ctx_str)


# ---------------------------------------------------------------------------
# 20. No hallucinated facts (system instruction check)
# ---------------------------------------------------------------------------


class TestNoHallucinatedFacts(unittest.TestCase):

    def test_system_instruction_forbids_fabrication(self):
        provider = FakeProvider(response="ok")
        sm = SessionManager()
        asst = BusinessAssistant(provider=provider, session_manager=sm)
        session = asst.new_session()
        req = BusinessRequest(
            task_type=BusinessTaskType.GENERAL_ANALYSIS,
            user_request="What is our revenue?",
        )
        asst.analyze(session.session_id, req)
        all_content = str(provider.last_messages)
        self.assertIn("NEVER fabricate", all_content)
        self.assertIn("NEVER pretend", all_content)


# ---------------------------------------------------------------------------
# 21. ActionItem generation/parsing
# ---------------------------------------------------------------------------


class TestActionItemStructure(unittest.TestCase):

    def test_action_item_to_context(self):
        ai = ActionItem(
            title="Launch marketing campaign",
            description="Run Q3 digital marketing campaign.",
            priority=Priority.HIGH,
            dependencies=["Budget approval"],
            expected_outcome="20% lead increase",
        )
        ctx = ai.to_context()
        self.assertIn("HIGH", ctx)
        self.assertIn("Launch marketing campaign", ctx)
        self.assertIn("Budget approval", ctx)
        self.assertIn("20% lead increase", ctx)

    def test_action_item_minimal(self):
        ai = ActionItem(title="Do something")
        ctx = ai.to_context()
        self.assertIn("MEDIUM", ctx)
        self.assertIn("Do something", ctx)


# ---------------------------------------------------------------------------
# 22. Recommendation structure
# ---------------------------------------------------------------------------


class TestRecommendationStructure(unittest.TestCase):

    def test_recommendation_to_context(self):
        rec = Recommendation(
            recommendation="Expand to European markets",
            rationale="Market research shows demand.",
            expected_impact="30% revenue growth",
            effort="High — requires localization",
            risk="Regulatory compliance challenges",
        )
        ctx = rec.to_context()
        self.assertIn("Expand to European markets", ctx)
        self.assertIn("Market research", ctx)
        self.assertIn("30% revenue growth", ctx)
        self.assertIn("High", ctx)
        self.assertIn("Regulatory", ctx)

    def test_recommendation_minimal(self):
        rec = Recommendation(recommendation="Cut costs")
        ctx = rec.to_context()
        self.assertIn("Cut costs", ctx)


# ---------------------------------------------------------------------------
# 23. Risk structure
# ---------------------------------------------------------------------------


class TestRiskStructure(unittest.TestCase):

    def test_business_plan_risks(self):
        plan = BusinessPlan(
            objective="Grow revenue",
            risks=[
                "Market downturn",
                "Increased competition",
                "Talent shortage",
            ],
        )
        ctx = plan.to_context()
        self.assertIn("Market downturn", ctx)
        self.assertIn("competition", ctx)
        self.assertIn("talent", ctx.lower())


# ---------------------------------------------------------------------------
# 24. Priority validation
# ---------------------------------------------------------------------------


class TestPriorityValidation(unittest.TestCase):

    def test_all_priority_values(self):
        self.assertEqual(Priority.LOW.value, "low")
        self.assertEqual(Priority.MEDIUM.value, "medium")
        self.assertEqual(Priority.HIGH.value, "high")
        self.assertEqual(Priority.CRITICAL.value, "critical")

    def test_priority_ordering(self):
        priorities = [Priority.LOW, Priority.MEDIUM, Priority.HIGH, Priority.CRITICAL]
        values = [p.value for p in priorities]
        self.assertEqual(
            values, ["low", "medium", "high", "critical"]
        )


# ---------------------------------------------------------------------------
# Additional: BusinessTaskType enum
# ---------------------------------------------------------------------------


class TestBusinessTaskTypeEnum(unittest.TestCase):

    def test_all_task_types_exist(self):
        expected = [
            "general_analysis", "company_analysis", "market_analysis",
            "sales", "lead_analysis", "customer_support", "marketing",
            "operations", "finance", "strategy", "kpi_analysis",
            "action_plan",
        ]
        actual = [t.value for t in BusinessTaskType]
        for e in expected:
            self.assertIn(e, actual)


# ---------------------------------------------------------------------------
# Additional: BusinessContext to_context_string
# ---------------------------------------------------------------------------


class TestBusinessContextString(unittest.TestCase):

    def test_full_context(self):
        ctx = BusinessContext(
            company_name="TestCo",
            industry="SaaS",
            company_description="B2B SaaS platform.",
            target_market="Mid-market companies",
            target_customer="CTOs and VPs of Engineering",
            products_services="Project management tool",
            business_goals="Reach $5M ARR",
            constraints="Limited budget, small team",
            competitors="Jira, Asana, Monday",
            current_problem="High churn rate",
            available_resources="15 engineers, $500K budget",
        )
        s = ctx.to_context_string()
        self.assertIn("TestCo", s)
        self.assertIn("SaaS", s)
        self.assertIn("Jira", s)
        self.assertIn("churn", s.lower())

    def test_empty_context(self):
        ctx = BusinessContext()
        self.assertEqual(ctx.to_context_string(), "")


# ---------------------------------------------------------------------------
# Additional: BusinessPlan
# ---------------------------------------------------------------------------


class TestBusinessPlan(unittest.TestCase):

    def test_full_plan(self):
        plan = BusinessPlan(
            objective="Reduce churn by 50%",
            current_state="Churn is 8% monthly",
            key_problems=["Poor onboarding", "Lack of support"],
            strategy="Invest in customer success",
            priorities=["Onboarding redesign", "Support team expansion"],
            action_items=[
                ActionItem(
                    title="Redesign onboarding",
                    priority=Priority.HIGH,
                    expected_outcome="40% faster activation",
                ),
                ActionItem(
                    title="Hire 2 support reps",
                    priority=Priority.MEDIUM,
                ),
            ],
            risks=["Budget constraints", "Hiring timeline"],
            dependencies=["Budget approval"],
            success_metrics=["Churn < 4%", "NPS > 50"],
        )
        ctx = plan.to_context()
        self.assertIn("Reduce churn", ctx)
        self.assertIn("onboarding", ctx.lower())
        self.assertIn("HIGH", ctx)
        self.assertIn("NPS", ctx)

    def test_minimal_plan(self):
        plan = BusinessPlan(objective="Test")
        ctx = plan.to_context()
        self.assertIn("Test", ctx)


# ---------------------------------------------------------------------------
# Additional: Session not found
# ---------------------------------------------------------------------------


class TestSessionNotFound(unittest.TestCase):

    def test_analyze_nonexistent_session(self):
        asst = _make_assistant()
        req = BusinessRequest(
            task_type=BusinessTaskType.GENERAL_ANALYSIS,
            user_request="test",
        )
        with self.assertRaises(SessionNotFoundError):
            asst.analyze("nonexistent", req)


# ---------------------------------------------------------------------------
# Additional: Chat convenience method
# ---------------------------------------------------------------------------


class TestChatConvenience(unittest.TestCase):

    def test_chat_method(self):
        asst = _make_assistant(response="Quick analysis.")
        session = _session_for(asst)
        resp = asst.chat(
            session.session_id,
            "Quick question about sales.",
            task_type=BusinessTaskType.SALES,
        )
        self.assertEqual(resp.response, "Quick analysis.")
        self.assertEqual(resp.task_type, BusinessTaskType.SALES)

    def test_chat_default_task_type(self):
        asst = _make_assistant()
        session = _session_for(asst)
        resp = asst.chat(session.session_id, "General question.")
        self.assertEqual(
            resp.task_type, BusinessTaskType.GENERAL_ANALYSIS
        )


# ---------------------------------------------------------------------------
# Additional: History persistence across turns
# ---------------------------------------------------------------------------


class TestMultiTurnBusiness(unittest.TestCase):

    def test_multiple_turns(self):
        asst = _make_assistant(response="reply")
        session = _session_for(asst)
        for i in range(3):
            req = BusinessRequest(
                task_type=BusinessTaskType.STRATEGY,
                user_request=f"strategy question {i}",
            )
            asst.analyze(session.session_id, req)
        self.assertEqual(len(session.history), 6)

    def test_history_in_context(self):
        provider = FakeProvider(response="reply")
        sm = SessionManager()
        asst = BusinessAssistant(provider=provider, session_manager=sm)
        session = asst.new_session()
        req = BusinessRequest(
            task_type=BusinessTaskType.SALES,
            user_request="first question",
        )
        asst.analyze(session.session_id, req)
        req2 = BusinessRequest(
            task_type=BusinessTaskType.SALES,
            user_request="follow-up question",
        )
        asst.analyze(session.session_id, req2)
        all_content = str(provider.last_messages)
        self.assertIn("first question", all_content)
        self.assertIn("follow-up question", all_content)


if __name__ == "__main__":
    unittest.main()
