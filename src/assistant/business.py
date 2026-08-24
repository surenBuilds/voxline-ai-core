"""
BusinessAssistant — business intelligence layer for Voxline.

Architecture:
    User request
        → SessionManager (get/create BUSINESS session)
        → BusinessAssistant (validate, build context, call provider)
        → ContextBuilder (assemble memory, history, business context)
        → AIProvider.chat()  (generate response)
        → Store messages in session
        → Optionally persist business memory
        → Return BusinessResponse

BusinessAssistant NEVER uses TextGenerator or QwenProvider directly.
All intelligence flows through AIProvider, which is replaceable.

Security boundary:
    BusinessAssistant is an ANALYSIS layer only.
    It must NOT execute shell commands, modify files, send emails,
    make financial transactions, or access arbitrary external systems.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.assistant.context import ContextBuilder, Context
from src.assistant.session import Session, SessionManager, SessionMode
from src.errors import (
    SessionNotFoundError,
    ProviderError,
    VoxlineError,
)
from src.memory.memory import MemoryStore
from src.providers.base import AIProvider, GenerationConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class BusinessTaskType(Enum):
    """Business analysis task types."""
    GENERAL_ANALYSIS = "general_analysis"
    COMPANY_ANALYSIS = "company_analysis"
    MARKET_ANALYSIS = "market_analysis"
    SALES = "sales"
    LEAD_ANALYSIS = "lead_analysis"
    CUSTOMER_SUPPORT = "customer_support"
    MARKETING = "marketing"
    OPERATIONS = "operations"
    FINANCE = "finance"
    STRATEGY = "strategy"
    KPI_ANALYSIS = "kpi_analysis"
    ACTION_PLAN = "action_plan"


class Priority(Enum):
    """Priority level for action items and recommendations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Structured business models
# ---------------------------------------------------------------------------


@dataclass
class KPI:
    """Key Performance Indicator."""
    name: str
    value: Optional[float] = None
    target: Optional[float] = None
    unit: Optional[str] = None
    period: Optional[str] = None

    def to_context(self) -> str:
        parts = [f"  {self.name}:"]
        if self.value is not None:
            val_str = f"{self.value}"
            if self.unit:
                val_str += f" {self.unit}"
            parts.append(f"    Value: {val_str}")
        else:
            parts.append("    Value: NOT PROVIDED")
        if self.target is not None:
            tgt_str = f"{self.target}"
            if self.unit:
                tgt_str += f" {self.unit}"
            parts.append(f"    Target: {tgt_str}")
        else:
            parts.append("    Target: NOT PROVIDED")
        if self.period:
            parts.append(f"    Period: {self.period}")
        return "\n".join(parts)


@dataclass
class ActionItem:
    """A single actionable step."""
    title: str
    description: str = ""
    priority: Priority = Priority.MEDIUM
    dependencies: List[str] = field(default_factory=list)
    expected_outcome: str = ""

    def to_context(self) -> str:
        lines = [
            f"  [{self.priority.value.upper()}] {self.title}",
        ]
        if self.description:
            lines.append(f"    {self.description}")
        if self.dependencies:
            lines.append(f"    Depends on: {', '.join(self.dependencies)}")
        if self.expected_outcome:
            lines.append(f"    Expected outcome: {self.expected_outcome}")
        return "\n".join(lines)


@dataclass
class Recommendation:
    """A strategic recommendation."""
    recommendation: str
    rationale: str = ""
    expected_impact: str = ""
    effort: str = ""
    risk: str = ""

    def to_context(self) -> str:
        lines = [f"  - {self.recommendation}"]
        if self.rationale:
            lines.append(f"    Rationale: {self.rationale}")
        if self.expected_impact:
            lines.append(f"    Expected impact: {self.expected_impact}")
        if self.effort:
            lines.append(f"    Effort: {self.effort}")
        if self.risk:
            lines.append(f"    Risk: {self.risk}")
        return "\n".join(lines)


@dataclass
class BusinessPlan:
    """Structured business plan."""
    objective: str = ""
    current_state: str = ""
    key_problems: List[str] = field(default_factory=list)
    strategy: str = ""
    priorities: List[str] = field(default_factory=list)
    action_items: List[ActionItem] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    success_metrics: List[str] = field(default_factory=list)

    def to_context(self) -> str:
        sections = []
        if self.objective:
            sections.append(f"Objective: {self.objective}")
        if self.current_state:
            sections.append(f"Current state: {self.current_state}")
        if self.key_problems:
            sections.append("Key problems:")
            for p in self.key_problems:
                sections.append(f"  - {p}")
        if self.strategy:
            sections.append(f"Strategy: {self.strategy}")
        if self.priorities:
            sections.append("Priorities:")
            for p in self.priorities:
                sections.append(f"  - {p}")
        if self.action_items:
            sections.append("Action items:")
            for ai in self.action_items:
                sections.append(ai.to_context())
        if self.risks:
            sections.append("Risks:")
            for r in self.risks:
                sections.append(f"  - {r}")
        if self.success_metrics:
            sections.append("Success metrics:")
            for m in self.success_metrics:
                sections.append(f"  - {m}")
        return "\n".join(sections)


@dataclass
class BusinessContext:
    """Typed business context for the assistant."""
    company_name: Optional[str] = None
    industry: Optional[str] = None
    company_description: Optional[str] = None
    target_market: Optional[str] = None
    target_customer: Optional[str] = None
    products_services: Optional[str] = None
    business_goals: Optional[str] = None
    constraints: Optional[str] = None
    current_kpis: Optional[List[KPI]] = None
    competitors: Optional[str] = None
    current_problem: Optional[str] = None
    available_resources: Optional[str] = None

    def to_context_string(self) -> str:
        """Format business context for injection into provider messages."""
        sections = []
        if self.company_name:
            sections.append(f"Company: {self.company_name}")
        if self.industry:
            sections.append(f"Industry: {self.industry}")
        if self.company_description:
            sections.append(f"Description: {self.company_description}")
        if self.target_market:
            sections.append(f"Target market: {self.target_market}")
        if self.target_customer:
            sections.append(f"Target customer: {self.target_customer}")
        if self.products_services:
            sections.append(f"Products/Services: {self.products_services}")
        if self.business_goals:
            sections.append(f"Business goals: {self.business_goals}")
        if self.constraints:
            sections.append(f"Constraints: {self.constraints}")
        if self.competitors:
            sections.append(f"Competitors: {self.competitors}")
        if self.current_problem:
            sections.append(f"Current problem: {self.current_problem}")
        if self.available_resources:
            sections.append(f"Available resources: {self.available_resources}")
        if self.current_kpis:
            sections.append("Current KPIs:")
            for kpi in self.current_kpis:
                sections.append(kpi.to_context())
        return "\n".join(sections)


@dataclass
class BusinessRequest:
    """Structured business request."""
    task_type: BusinessTaskType
    user_request: str
    business_context: Optional[BusinessContext] = None
    desired_output: Optional[str] = None
    language: Optional[str] = None


@dataclass
class BusinessResponse:
    """Structured business response."""
    response: str
    task_type: BusinessTaskType
    session_id: str
    provider_id: str
    model_id: str
    language: Optional[str] = None
    recommendations: Optional[List[Recommendation]] = None
    action_items: Optional[List[ActionItem]] = None
    risks: Optional[List[str]] = None
    assumptions: Optional[List[str]] = None
    confidence: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# System instructions
# ---------------------------------------------------------------------------

_BUSINESS_SYSTEM_INSTRUCTION = (
    "You are a professional business analyst and strategic assistant.\n\n"
    "CORE BEHAVIOR:\n"
    "- Reason from the information provided to you.\n"
    "- Distinguish KNOWN FACTS from ASSUMPTIONS from RECOMMENDATIONS.\n"
    "- Explicitly identify MISSING INFORMATION.\n"
    "- NEVER fabricate company facts or claim to have performed external "
    "research unless a tool was actually used.\n"
    "- NEVER pretend to have real-time market data.\n\n"
    "OUTPUT:\n"
    "- Provide actionable, prioritized recommendations.\n"
    "- Identify risks and tradeoffs.\n"
    "- When appropriate, use structured output:\n"
    "  EXECUTIVE SUMMARY, KEY FINDINGS, ANALYSIS, RECOMMENDATIONS, "
    "PRIORITIES, ACTION PLAN, RISKS, MISSING INFORMATION.\n"
    "- Do NOT force this structure on every casual business conversation.\n"
    "- Ask for missing critical information when necessary.\n\n"
    "LANGUAGE:\n"
    "- Respond in the same language as the user request.\n"
    "- Do NOT translate unless explicitly requested.\n"
    "- Use professional business terminology appropriate to the language.\n\n"
    "IMPORTANT:\n"
    "- Do NOT execute any actions. You are an analysis and advisory layer only.\n"
    "- Do NOT send emails, modify files, or access external systems.\n"
    "- You can RECOMMEND actions but must not EXECUTE them."
)

_TASK_TYPE_INSTRUCTIONS: Dict[BusinessTaskType, str] = {
    BusinessTaskType.GENERAL_ANALYSIS: (
        "Provide a balanced analysis of the business situation. "
        "Identify key facts, assumptions, and recommendations."
    ),
    BusinessTaskType.COMPANY_ANALYSIS: (
        "Analyze the company's strengths, weaknesses, opportunities, and threats. "
        "Consider market position, capabilities, and competitive landscape."
    ),
    BusinessTaskType.MARKET_ANALYSIS: (
        "Analyze the market environment. Consider market size, trends, "
        "competition, customer segments, and growth opportunities."
    ),
    BusinessTaskType.SALES: (
        "Focus on sales strategy, pipeline, conversion, and revenue optimization. "
        "Provide actionable sales recommendations."
    ),
    BusinessTaskType.LEAD_ANALYSIS: (
        "Analyze lead quality, sources, conversion patterns, and "
        "provide recommendations for lead generation and nurturing."
    ),
    BusinessTaskType.CUSTOMER_SUPPORT: (
        "Analyze customer support operations. Consider response times, "
        "satisfaction, common issues, and process improvements."
    ),
    BusinessTaskType.MARKETING: (
        "Analyze marketing strategy, channels, ROI, and brand positioning. "
        "Provide actionable marketing recommendations."
    ),
    BusinessTaskType.OPERATIONS: (
        "Analyze operational efficiency, processes, resource allocation, "
        "and provide recommendations for optimization."
    ),
    BusinessTaskType.FINANCE: (
        "Analyze financial health, revenue, costs, margins, and "
        "provide financial recommendations. "
        "NEVER invent financial figures. State clearly if data is missing."
    ),
    BusinessTaskType.STRATEGY: (
        "Provide strategic analysis and recommendations. "
        "Consider long-term positioning, competitive advantages, "
        "and sustainable growth."
    ),
    BusinessTaskType.KPI_ANALYSIS: (
        "Analyze the provided KPIs. Compare values to targets where available. "
        "Identify underperforming metrics and suggest improvement actions. "
        "NEVER invent KPI values. State clearly if a value or target is missing."
    ),
    BusinessTaskType.ACTION_PLAN: (
        "Transform the business problem into a structured action plan. "
        "Prioritize action items and identify dependencies."
    ),
}


# ---------------------------------------------------------------------------
# BusinessAssistant
# ---------------------------------------------------------------------------


class BusinessAssistant:
    """
    Business intelligence assistant.

    Orchestrates session management, context construction, provider
    invocation, and optional business memory persistence.

    Operates ONLY under SessionMode.BUSINESS.
    The provider is the only source of intelligence — BusinessAssistant
    contains no model logic.
    """

    _MEMORY_KEYWORDS = frozenset({
        "company", "product", "goal", "target", "revenue",
        "customer", "market", "strategy", "budget", "plan",
        "kpi", "metric", "competitor", "constraint",
    })

    def __init__(
        self,
        provider: AIProvider,
        session_manager: SessionManager,
        context_builder: Optional[ContextBuilder] = None,
        memory_store: Optional[MemoryStore] = None,
        max_history: int = 20,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ):
        self.provider = provider
        self.session_manager = session_manager
        self.memory_store = memory_store
        self.max_history = max_history
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._context_builder = context_builder or ContextBuilder(
            memory_store=memory_store,
            max_history=max_history,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        session_id: str,
        request: BusinessRequest,
        persist_memory: Optional[bool] = None,
    ) -> BusinessResponse:
        """
        Process a structured business request.

        Args:
            session_id: Session to use (must exist, must be BUSINESS mode).
            request: Structured business request.
            persist_memory: Force memory persistence. None = auto-detect.

        Returns:
            BusinessResponse with text, structured data, and metadata.

        Raises:
            SessionNotFoundError: Session does not exist.
            ValueError: Invalid request.
            ProviderError: Provider failed to generate.
        """
        self._validate_request(request)
        session = self._get_session(session_id)
        self._assert_business_mode(session)

        biz_ctx_str = self._build_business_context_string(
            request.business_context
        )
        task_instruction = _TASK_TYPE_INSTRUCTIONS.get(
            request.task_type, ""
        )

        ctx = self._context_builder.build(
            session=session,
            user_message=request.user_request,
            mode_instructions=(
                _BUSINESS_SYSTEM_INSTRUCTION + "\n\n" + task_instruction
            ),
            business_context=biz_ctx_str if biz_ctx_str else None,
        )

        text = self._call_provider(ctx)

        session.add_message("user", request.user_request)
        session.add_message("assistant", text)

        self._maybe_persist_memory(
            request.user_request, request.business_context, persist_memory
        )

        return BusinessResponse(
            response=text,
            task_type=request.task_type,
            session_id=session.session_id,
            provider_id=self.provider.get_model_info().provider_id,
            model_id=self.provider.get_model_info().model_id,
            language=request.language,
            metadata={
                "history_length": len(session.history),
                "has_business_context": request.business_context is not None,
            },
        )

    def chat(
        self,
        session_id: str,
        message: str,
        task_type: BusinessTaskType = BusinessTaskType.GENERAL_ANALYSIS,
        persist_memory: Optional[bool] = None,
    ) -> BusinessResponse:
        """
        Convenience method for quick business queries without full
        BusinessRequest construction.

        Args:
            session_id: Session to use.
            message: User's business question.
            task_type: Business task type (defaults to GENERAL_ANALYSIS).
            persist_memory: Force memory persistence.

        Returns:
            BusinessResponse.
        """
        request = BusinessRequest(
            task_type=task_type,
            user_request=message,
        )
        return self.analyze(session_id, request, persist_memory)

    def new_session(
        self,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Session:
        """Create and return a new BUSINESS session."""
        return self.session_manager.create(
            SessionMode.BUSINESS, metadata=metadata
        )

    # ------------------------------------------------------------------
    # Business context
    # ------------------------------------------------------------------

    @staticmethod
    def _build_business_context_string(
        biz_ctx: Optional[BusinessContext],
    ) -> str:
        if biz_ctx is None:
            return ""
        return biz_ctx.to_context_string()

    # ------------------------------------------------------------------
    # Provider interaction
    # ------------------------------------------------------------------

    def _call_provider(self, ctx: Context) -> str:
        config = GenerationConfig(
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        try:
            import asyncio

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.provider.chat(ctx.messages, config),
                    )
                    return future.result(timeout=60)
            else:
                return asyncio.run(self.provider.chat(ctx.messages, config))
        except Exception as exc:
            raise ProviderError(f"Provider failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_request(request: BusinessRequest) -> None:
        if not isinstance(request, BusinessRequest):
            raise ValueError("Request must be a BusinessRequest instance.")
        if not isinstance(request.task_type, BusinessTaskType):
            raise ValueError("task_type must be a BusinessTaskType enum value.")
        if not request.user_request or not request.user_request.strip():
            raise ValueError("user_request must be a non-empty string.")

    def _get_session(self, session_id: str) -> Session:
        session = self.session_manager.get(session_id)
        if session is None:
            raise SessionNotFoundError(
                f"Session '{session_id}' does not exist."
            )
        return session

    @staticmethod
    def _assert_business_mode(session: Session) -> None:
        if session.mode != SessionMode.BUSINESS:
            raise ValueError(
                f"BusinessAssistant requires SessionMode.BUSINESS, "
                f"got SessionMode.{session.mode.name}."
            )

    # ------------------------------------------------------------------
    # Memory persistence
    # ------------------------------------------------------------------

    def _maybe_persist_memory(
        self,
        message: str,
        biz_ctx: Optional[BusinessContext],
        persist_memory: Optional[bool],
    ) -> None:
        if self.memory_store is None:
            return

        should_persist = persist_memory is True or (
            persist_memory is None and self._looks_like_business_memory(
                message, biz_ctx
            )
        )

        if should_persist:
            self._store_business_memory(message, biz_ctx)

    def _looks_like_business_memory(
        self, message: str, biz_ctx: Optional[BusinessContext]
    ) -> bool:
        lower = message.lower()
        has_keyword = any(kw in lower for kw in self._MEMORY_KEYWORDS)
        has_ctx = biz_ctx is not None and biz_ctx.company_name is not None
        return has_keyword or has_ctx

    def _store_business_memory(
        self, content: str, biz_ctx: Optional[BusinessContext]
    ) -> None:
        try:
            tags = ["business"]
            if biz_ctx and biz_ctx.company_name:
                tags.append(biz_ctx.company_name.lower())
            self.memory_store.add_memory(
                content=content,
                memory_type="semantic",
                source="conversation",
                keywords=content.split()[:5],
                tags=tags,
            )
            logger.debug("Persisted business memory: %s", content[:60])
        except Exception:
            logger.debug("Business memory persistence failed", exc_info=True)
