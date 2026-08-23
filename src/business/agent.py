"""A safe, local-first foundation for business planning in Voxline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import uuid4

from src.memory.memory import MemoryStore


@dataclass
class BusinessPlanStep:
    """One concrete, reviewable action in a business plan."""

    id: str
    title: str
    description: str
    status: str = "pending"


@dataclass
class BusinessPlan:
    """A local plan created for a business goal."""

    id: str
    goal: str
    created_at: str
    steps: List[BusinessPlanStep]
    context: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class BusinessAgent:
    """Local-first business assistant with persistent knowledge and plans."""

    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store
        self.plans: Dict[str, BusinessPlan] = {}

    def remember(self, content: str, tags: Optional[List[str]] = None) -> str:
        """Store a business fact locally for future retrieval."""
        clean_content = content.strip()
        if not clean_content:
            raise ValueError("Business knowledge cannot be empty")
        return self.memory_store.add_memory(
            content=clean_content,
            memory_type="semantic",
            source="business_agent",
            keywords=self._keywords(clean_content),
            tags=["business", *(tags or [])],
        )

    def search_knowledge(self, query: str, limit: int = 5) -> List[Dict]:
        """Retrieve locally stored business knowledge."""
        if not query.strip():
            return []
        return [item.to_dict() for item in self.memory_store.search_memories(query, limit=limit)]

    def create_plan(self, goal: str, context: Optional[str] = None) -> BusinessPlan:
        """Create a transparent baseline plan which a user may edit or approve."""
        clean_goal = goal.strip()
        if not clean_goal:
            raise ValueError("Goal cannot be empty")

        steps = self._steps_for_goal(clean_goal)
        plan = BusinessPlan(
            id=f"business_plan_{uuid4().hex[:12]}",
            goal=clean_goal,
            context=context.strip() if context else None,
            created_at=datetime.now(timezone.utc).isoformat(),
            steps=[
                BusinessPlanStep(id=f"step_{index}", title=title, description=description)
                for index, (title, description) in enumerate(steps, start=1)
            ],
        )
        self.plans[plan.id] = plan
        self.remember(f"Business goal: {clean_goal}", tags=["goal"])
        return plan

    def get_plan(self, plan_id: str) -> Optional[BusinessPlan]:
        return self.plans.get(plan_id)

    @staticmethod
    def _keywords(text: str) -> List[str]:
        return sorted({word.strip(".,:;!?()[]{}\"").lower() for word in text.split() if len(word) > 2})[:20]

    @staticmethod
    def _steps_for_goal(goal: str) -> List[tuple[str, str]]:
        lower_goal = goal.lower()
        if any(word in lower_goal for word in ("marketing", "մարքեթ", "գովազդ", "content")):
            middle = ("Հետազոտել լսարանը", "Սահմանել թիրախային հաճախորդին, նրա խնդիրը և մրցակիցների տարբերակումը.")
        elif any(word in lower_goal for word in ("sales", "վաճառ", "lead", "հաճախորդ")):
            middle = ("Կառուցել վաճառքի ուղին", "Սահմանել առաջարկը, լիդի աղբյուրները, հաղորդագրությունը և վաճառքի չափումները.")
        elif any(word in lower_goal for word in ("product", "ապրանք", "ծառայ", "պրոդուկտ")):
            middle = ("Հստակեցնել արժեքային առաջարկը", "Նկարագրել հաճախորդի խնդիրը, լուծումը, MVP-ի սահմանները և հաջողության չափանիշը.")
        else:
            middle = ("Վերլուծել հնարավորությունը", "Սահմանել հաճախորդին, շուկան, առաջարկը, ռիսկերը և չափելի արդյունքը.")

        return [
            ("Սահմանել հաջողության չափանիշը", f"Գրել, թե ինչ արդյունք է նշանակում հաջողություն այս նպատակի համար՝ {goal}."),
            middle,
            ("Կազմել 7-օրյա գործողությունների պլան", "Բաժանել աշխատանքը փոքր, պատասխանատու և ժամկետով գործողությունների."),
            ("Չափել և վերանայել", "Ընտրել 2-3 չափիչ, շաբաթվա վերջում գրանցել արդյունքը և թարմացնել պլանը."),
        ]
