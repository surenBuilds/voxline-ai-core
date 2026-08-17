"""
Autonomous agent for Voxline AI Core

Orchestrates:
- Language model
- Memory
- Planning
- Tools
- Execution loop
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import time


class AgentState(Enum):
    """Agent execution state."""

    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    OBSERVING = "observing"
    REVISING = "revising"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ExecutionLog:
    """Execution log entry."""

    timestamp: float
    state: AgentState
    action: str
    result: Optional[str] = None
    error: Optional[str] = None


class AutonomousAgent:
    """
    Autonomous agent combining model, memory, planning, and tools.
    """

    def __init__(
        self,
        model,
        tokenizer,
        memory_store=None,
        tool_registry=None,
        reasoning_engine=None,
        device: str = "cpu",
        max_iterations: int = 10,
        timeout: int = 300,
    ):
        """
        Initialize autonomous agent.

        Args:
            model: Language model
            tokenizer: Tokenizer
            memory_store: Memory store
            tool_registry: Available tools
            reasoning_engine: Planning and reasoning
            device: Execution device
            max_iterations: Max execution iterations
            timeout: Execution timeout in seconds
        """
        from src.api.chat import ConversationalAI
        from src.memory.memory import MemoryStore
        from src.tools.tools import ToolRegistry
        from src.planner.reasoning import ReasoningEngine

        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Initialize subsystems
        self.memory_store = memory_store or MemoryStore()
        self.chat = ConversationalAI(
            model, tokenizer, self.memory_store, device
        )
        self.tools = tool_registry or ToolRegistry()
        self.reasoning = reasoning_engine or ReasoningEngine()

        # Agent parameters
        self.max_iterations = max_iterations
        self.timeout = timeout

        # Execution state
        self.state = AgentState.IDLE
        self.current_goal = None
        self.execution_logs: List[ExecutionLog] = []
        self.task_counter = 0

    def set_goal(self, goal: str):
        """Set agent goal."""
        self.current_goal = goal
        self.state = AgentState.IDLE
        self.execution_logs = []

    def run(self, goal: str) -> Dict[str, Any]:
        """
        Execute agent loop.

        Args:
            goal: Task goal

        Returns:
            Execution result
        """
        self.set_goal(goal)
        start_time = time.time()

        try:
            # Plan phase
            self._log_state(AgentState.PLANNING, f"Planning for goal: {goal}")
            self.state = AgentState.PLANNING
            plan = self.reasoning.create_execution_plan(goal)

            # Execution loop
            iteration = 0
            while iteration < self.max_iterations:
                if time.time() - start_time > self.timeout:
                    self._log_state(AgentState.FAILED, "Timeout exceeded")
                    self.state = AgentState.FAILED
                    break

                iteration += 1

                # Get next step
                next_step = self.reasoning.planner.get_next_step(plan.id)
                if not next_step:
                    break

                # Execute step
                self._log_state(AgentState.EXECUTING, f"Executing: {next_step.description}")
                self.state = AgentState.EXECUTING

                result = self._execute_step(next_step)

                # Observe result
                self._log_state(AgentState.OBSERVING, f"Observing result")
                self.state = AgentState.OBSERVING

                if "error" in result:
                    self.reasoning.planner.mark_plan_failed(plan.id)
                    self.reasoning.planner.update_step_error(
                        plan.id, next_step.id, result.get("error", "Unknown error")
                    )

                    # Attempt revision
                    if self.reasoning.should_revise_plan(plan.id):
                        self._log_state(AgentState.REVISING, "Revising plan")
                        self.state = AgentState.REVISING
                        # Would replan here
                    else:
                        self.state = AgentState.FAILED
                        break
                else:
                    self.reasoning.planner.update_step_result(
                        plan.id, next_step.id, str(result)
                    )

            # Verify completion
            self._log_state(AgentState.VERIFYING, "Verifying completion")
            self.state = AgentState.VERIFYING

            if self.reasoning.planner.get_next_step(plan.id) is None:
                self._log_state(AgentState.COMPLETED, "Task completed successfully")
                self.state = AgentState.COMPLETED
                success = True
            else:
                self._log_state(AgentState.FAILED, "Task not fully completed")
                self.state = AgentState.FAILED
                success = False

            return {
                "success": success,
                "goal": goal,
                "plan": plan.to_dict(),
                "execution_logs": self._format_logs(),
                "duration": time.time() - start_time,
            }

        except Exception as e:
            self._log_state(AgentState.FAILED, f"Error: {str(e)}")
            self.state = AgentState.FAILED
            return {
                "success": False,
                "goal": goal,
                "error": str(e),
                "execution_logs": self._format_logs(),
                "duration": time.time() - start_time,
            }

    def _execute_step(self, step) -> Dict[str, Any]:
        """
        Execute single step.

        Args:
            step: Step to execute

        Returns:
            Step result
        """
        try:
            # Determine action type and execute
            if step.action == "read_file":
                result = self.tools.execute_tool("read_file", **step.action_params)
                return {"result": result}
            elif step.action == "write_file":
                result = self.tools.execute_tool("write_file", **step.action_params)
                return result
            elif step.action == "chat":
                result = self.chat.chat(step.action_params.get("message", ""))
                return {"result": result}
            elif step.action == "search_memory":
                result = self.chat.search_memory(step.action_params.get("query", ""))
                return {"result": result}
            else:
                return {"error": f"Unknown action: {step.action}"}
        except Exception as e:
            return {"error": str(e)}

    def _log_state(self, state: AgentState, action: str):
        """Log state transition."""
        log = ExecutionLog(
            timestamp=time.time(),
            state=state,
            action=action,
        )
        self.execution_logs.append(log)
        print(f"[{state.value}] {action}")

    def _format_logs(self) -> List[Dict[str, Any]]:
        """Format execution logs."""
        return [
            {
                "timestamp": log.timestamp,
                "state": log.state.value,
                "action": log.action,
                "error": log.error,
            }
            for log in self.execution_logs
        ]

    def get_state(self) -> AgentState:
        """Get current agent state."""
        return self.state

    def get_memory(self) -> MemoryStore:
        """Get memory store."""
        return self.memory_store

    def get_tools(self) -> List[str]:
        """Get available tools."""
        return list(self.tools.tools.keys())

    def add_tool(self, name: str, tool):
        """Add new tool."""
        self.tools.register(name, tool)

    def cancel(self):
        """Cancel current execution."""
        self.state = AgentState.FAILED
