import asyncio
import os
import sys
import textwrap
from pathlib import Path
from typing import List, Optional

from openai import OpenAI
from dotenv import load_dotenv

# Ensure server module is discoverable
sys.path.append(str(Path(__file__).parent / "server"))

try:
    from client import CopywritingEnv
    from models import CallToolAction
except ImportError:
    from server.client import CopywritingEnv
    from server.models import CallToolAction

load_dotenv()

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = HF_TOKEN or os.getenv("OPENAI_API_KEY")

ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
TASK_NAME = os.getenv("TASK_NAME", "copywriting")
BENCHMARK = os.getenv("BENCHMARK", "copywriting_env")

MAX_STEPS = 3
SUCCESS_THRESHOLD = 0.6

_SYSTEM = {
    "subject_line_rewrite": (
        "You are a marketing copywriter. Rewrite the given subject line to maximise open rates. "
        "Return ONLY the rewritten subject line — no explanation, no quotes, max 60 characters."
    ),
    "cold_email_draft": (
        "You are a marketing copywriter. Write the requested cold outreach email. "
        "Return ONLY the email body — no subject line header, no explanation."
    ),
    "ab_copy_judge": textwrap.dedent("""
        You are a B2B marketing expert judging campaign performance for CFO outreach.
        Format response exactly as:
        WINNER: [A or B]
        REASON 1: ...
        REASON 2: ...
        REASON 3: ...
    """).strip(),
}

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    safe_action = action.replace("\n", " ").replace("\r", "")[:120]
    error_val = error or "null"
    print(f"[STEP] step={step} action={safe_action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

def get_model_completion(client: OpenAI, tool_name: str, prompt: str) -> str:
    system_prompt = _SYSTEM.get(tool_name, "Complete the task.")
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=400,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[DEBUG] Model error: {e}", flush=True)
        return ""

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = CopywritingEnv(base_url=ENV_URL)

    rewards: List[float] = []
    steps_taken = 0
    score, success = 0.0, False

    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)

    try:
        await env.reset()
        tools = ["subject_line_rewrite", "cold_email_draft", "ab_copy_judge"]
        
        for step, tool_name in enumerate(tools, start=1):
            message = get_model_completion(client, tool_name, f"Perform the {tool_name} task.")
            action = CallToolAction(tool_name=tool_name, arguments={"candidate": message})
            result = await env.step(action)

            reward = float(result.reward or 0.0)
            done = bool(result.done)

            rewards.append(reward)
            steps_taken = step
            log_step(step, message, reward, done, None)

            if done:
                break

        score = min(max(sum(rewards) / MAX_STEPS, 0.0), 1.0) if MAX_STEPS > 0 else 0.0
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Execution error: {e}", flush=True)
    finally:
        try:
            await env.close()
        except Exception:
            pass
        log_end(success, steps_taken, score, rewards)

if __name__ == "__main__":
    asyncio.run(main())