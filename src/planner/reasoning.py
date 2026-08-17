"""
Reasoning and planning system

Implements structured task decomposition and planning:
- Goal analysis
- Task decomposition
- Plan generation
- Step execution
- Observation and evaluation
- Revision
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum


class PlanStatus(Enum):
    """Plan status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REVISED = "revised"


@dataclass
class Step:
    """Single step in a plan."""

    id: str
    description: str
    action: str  # e.g., "read_file", "run_code", "search_memory"
    action_params: Dict[str, Any]
    status: PlanStatus = PlanStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self):
        return {
            "id": self.id,
            "description": self.description,
            "action": self.action,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
        }


@dataclass
class Plan:
    """Execution plan."""

    id: str
    goal: str
    steps: List[Step]
    status: PlanStatus = PlanStatus.PENDING
    total_steps: int = 0
    completed_steps: int = 0

    def add_step(self, step: Step):
        """Add step to plan."""
        self.steps.append(step)
        self.total_steps += 1

    def mark_step_complete(self, step_id: str, result: str):
        """Mark step as complete."""
        for step in self.steps:
            if step.id == step_id:
                step.status = PlanStatus.COMPLETED
                step.result = result
                self.completed_steps += 1
                break

    def mark_step_failed(self, step_id: str, error: str):
        """Mark step as failed."""
        for step in self.steps:
            if step.id == step_id:
                step.status = PlanStatus.FAILED
                step.error = error
                break

    def get_progress(self) -> float:
        """Get completion progress 0-1."""
        if self.total_steps == 0:
            return 0.0
        return self.completed_steps / self.total_steps

    def to_dict(self):
        return {
            "id": self.id,
            "goal": self.goal,
            "status": self.status.value,
            "progress": self.get_progress(),
            "steps": [s.to_dict() for s in self.steps],
        }


class Planner:
    """Task planner and reasoning engine."""

    def __init__(self):
        """Initialize planner."""
        self.plans: Dict[str, Plan] = {}
        self.plan_counter = 0

    def create_plan(self, goal: str, steps_descriptions: List[str]) -> Plan:
        """
        Create execution plan.

        Args:
            goal: Overall goal
            steps_descriptions: List of step descriptions

        Returns:
            Plan object
        """
        plan_id = f"plan_{self.plan_counter}"
        self.plan_counter += 1

        plan = Plan(id=plan_id, goal=goal, steps=[])

        for i, desc in enumerate(steps_descriptions):
            step = Step(
                id=f"{plan_id}_step_{i}",
                description=desc,
                action="",  # To be determined during execution
                action_params={},
            )
            plan.add_step(step)

        self.plans[plan_id] = plan
        return plan

    def decompose_task(self, task: str) -> List[str]:
        """
        Decompose complex task into steps.

        Args:
            task: Task description

        Returns:
            List of step descriptions
        """
        # This would be powered by the language model in real implementation
        # For now, return empty list - to be implemented with agent
        return []

    def get_plan(self, plan_id: str) -> Optional[Plan]:
        """Get plan by ID."""
        return self.plans.get(plan_id)

    def update_step_result(self, plan_id: str, step_id: str, result: str):
        """Update step result."""
        plan = self.get_plan(plan_id)
        if plan:
            plan.mark_step_complete(step_id, result)

    def update_step_error(self, plan_id: str, step_id: str, error: str):
        """Update step error."""
        plan = self.get_plan(plan_id)
        if plan:
            plan.mark_step_failed(step_id, error)

    def get_next_step(self, plan_id: str) -> Optional[Step]:
        """Get next pending step."""
        plan = self.get_plan(plan_id)
        if plan:
            for step in plan.steps:
                if step.status == PlanStatus.PENDING:
                    return step
        return None

    def mark_plan_complete(self, plan_id: str):
        """Mark plan as complete."""
        plan = self.get_plan(plan_id)
        if plan:
            plan.status = PlanStatus.COMPLETED

    def mark_plan_failed(self, plan_id: str):
        """Mark plan as failed."""
        plan = self.get_plan(plan_id)
        if plan:
            plan.status = PlanStatus.FAILED


class ReasoningEngine:
    """Reasoning engine for inference and analysis."""

    def __init__(self):
        """Initialize reasoning engine."""
        self.planner = Planner()

    def analyze_goal(self, goal: str) -> Dict[str, Any]:
        """
        Analyze goal and identify requirements.

        Args:
            goal: Goal description

        Returns:
            Analysis result
        """
        return {
            "goal": goal,
            "complexity": "unknown",  # Would be determined by model
            "requires_tools": [],
            "requires_memory": False,
            "estimated_steps": 0,
        }

    def create_execution_plan(self, goal: str) -> Plan:
        """
        Create execution plan for goal.

        Args:
            goal: Goal description

        Returns:
            Execution plan
        """
        # Decompose into steps
        steps = self.planner.decompose_task(goal)

        # Create plan
        plan = self.planner.create_plan(goal, steps)

        return plan

    def evaluate_progress(self, plan_id: str) -> Dict[str, Any]:
        """
        Evaluate plan progress.

        Args:
            plan_id: Plan ID

        Returns:
            Progress evaluation
        """
        plan = self.planner.get_plan(plan_id)
        if not plan:
            return {"error": "Plan not found"}

        return {
            "plan_id": plan_id,
            "status": plan.status.value,
            "progress": plan.get_progress(),
            "completed_steps": plan.completed_steps,
            "total_steps": plan.total_steps,
        }

    def should_revise_plan(self, plan_id: str) -> bool:
        """
        Determine if plan should be revised.

        Args:
            plan_id: Plan ID

        Returns:
            Whether plan should be revised
        """
        plan = self.planner.get_plan(plan_id)
        if not plan:
            return False

        # Check if too many failures
        failed_count = sum(1 for s in plan.steps if s.status == PlanStatus.FAILED)
        return failed_count > len(plan.steps) * 0.3
