import json
from typing import Literal, Any, Annotated

from pydantic import BaseModel, Field, BeforeValidator, ConfigDict, model_validator

from openenv.core.env_server.types import Action, Observation
from openenv.core.env_server.mcp_types import CallToolObservation


def _coerce_to_dict(v: Any) -> Any:
    """Parse JSON strings into dicts — handles Gradio UI sending strings."""
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v
    return v


# accepts both dict and JSON string
JsonDict = Annotated[dict, BeforeValidator(_coerce_to_dict)]


class CopywritingObservation(CallToolObservation):
    model_config = ConfigDict(extra="allow")


class CallToolAction(Action):
    type: Literal["call_tool"] = "call_tool"
    tool_name: str
    arguments: Any = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def handle_json_string(cls, data: Any) -> Any:
        if isinstance(data, dict):
            args = data.get("arguments")
            if isinstance(args, str):
                try:
                    data["arguments"] = json.loads(args)
                except Exception:
                    pass
        return data


class GradeRequest(BaseModel):
    candidate: str = Field(..., description="The agent's text response to be graded.")


class GradeBreakdown(BaseModel):
    """Per-dimension scores — fields vary by task, unused ones are None."""
    # task 1
    length_score: float | None = None
    power_score: float | None = None
    sentiment_score: float | None = None
    char_count: int | None = None
    power_words_hit: list[str] | None = None
    vader_compound: float | None = None
    # task 2
    wc_score: float | None = None
    cta_score: float | None = None
    fk_score: float | None = None
    word_count: int | None = None
    cta_match: str | None = None
    flesch_ease: float | None = None
    # task 3
    choice_score: float | None = None
    reason_score: float | None = None
    kw_score: float | None = None
    chosen: str | None = None
    correct: str | None = None
    reasons_found: int | None = None
    keywords_hit: list[str] | None = None


class GradeResponse(BaseModel):
    task_id: str
    difficulty: str
    prompt: str
    reward: float = Field(..., ge=0.0, le=1.0)
    feedback: str
    breakdown: GradeBreakdown