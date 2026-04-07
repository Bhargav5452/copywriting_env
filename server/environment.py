from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from fastmcp import FastMCP

from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation, State

try:
    from .graders import grade_ab_judge, grade_cold_email, grade_subject_line
    from .tasks import TASKS
except ImportError:
    from graders import grade_ab_judge, grade_cold_email, grade_subject_line
    from tasks import TASKS


_REQUIRED_TOOLS = {"subject_line_rewrite", "cold_email_draft", "ab_copy_judge"}


class CopywritingEnvironment(MCPEnvironment):
    """Three-task marketing copywriting environment."""

    def __init__(self) -> None:
        mcp = FastMCP("copywriting_env")
        self._task1_output = ""
        self._completed_tools: set[str] = set()

        @mcp.tool
        def subject_line_rewrite(candidate: str) -> dict:
            """Rewrite a weak subject line. Graded on length, power words, sentiment."""
            self._completed_tools.add("subject_line_rewrite")
            self._task1_output = candidate
            task = TASKS["subject_line_rewrite"]
            return {"task_id": task["id"], "difficulty": task["difficulty"],
                    "prompt": task["prompt"], **grade_subject_line(candidate, task["ground_truth"])}

        @mcp.tool
        def cold_email_draft(candidate: str) -> dict:
            """Draft a cold email for a CFO. Graded on word count, CTA, readability."""
            self._completed_tools.add("cold_email_draft")
            task = TASKS["cold_email_draft"]
            prompt = task["prompt"].format(headline=self._task1_output or "No headline provided")
            return {"task_id": task["id"], "difficulty": task["difficulty"],
                    "prompt": prompt, **grade_cold_email(candidate, task["ground_truth"])}

        @mcp.tool
        def ab_copy_judge(candidate: str) -> dict:
            """Pick the winning A/B campaign variant with 3 reasons. Ground truth = B."""
            self._completed_tools.add("ab_copy_judge")
            task = TASKS["ab_copy_judge"]
            return {"task_id": task["id"], "difficulty": task["difficulty"],
                    "prompt": task["prompt"], **grade_ab_judge(candidate, task["ground_truth"])}

        super().__init__(mcp)
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._task1_output = ""
        self._completed_tools = set()
        return Observation(
            done=False,
            reward=0.0,
            metadata={"status": "ready",
                       "message": "Available tools: subject_line_rewrite | cold_email_draft | ab_copy_judge"},
        )

    def _step_impl(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        return Observation(
            done=False,
            reward=0.0,
            metadata={"error": f"Unknown action: {type(action).__name__}. Use CallToolAction."},
        )

    def _apply_tool_result(self, obs: Observation) -> Observation:
        """Pull reward/done out of the nested tool result onto the top-level obs."""
        if hasattr(obs, "result") and obs.result and hasattr(obs.result, "data"):
            data = obs.result.data
            if isinstance(data, dict):
                if "reward" in data:
                    obs.reward = float(data["reward"])
                if "done" in data:
                    obs.done = bool(data["done"])
        self._state.step_count += 1
        if self._completed_tools >= _REQUIRED_TOOLS:
            obs.done = True
        return obs

    def step(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        return self._apply_tool_result(super().step(action, timeout_s=timeout_s, **kwargs))

    async def step_async(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        return self._apply_tool_result(await super().step_async(action, timeout_s=timeout_s, **kwargs))

    @property
    def state(self) -> State:
        return self._state