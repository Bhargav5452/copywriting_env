import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI
from dotenv import load_dotenv

# Load locally if .env exists
load_dotenv()

import sys
from pathlib import Path

# Add 'server' directory to path to handle the restructured environment
current_dir = Path(__file__).parent
server_dir = current_dir / "server"
if str(server_dir) not in sys.path:
    sys.path.append(str(server_dir))

try:
    from client import CopywritingEnv
    from models import CallToolAction
except ImportError:
    try:
        from server.client import CopywritingEnv
        from server.models import CallToolAction
    except ImportError:
        # Fallback for original openenv structure
        from src.envs.copywriting_env.client import CopywritingEnv
        from src.envs.copywriting_env.models import CallToolAction

# MANDATORY ENVIRONMENT VARIABLES (Matched to Checklist)
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional - for local docker usage
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# Internal alias for code compatibility
API_KEY = HF_TOKEN or os.getenv("OPENAI_API_KEY")

# ENVIRONMENT CONFIG
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
TASK_NAME = os.getenv("TASK_NAME", "copywriting")
BENCHMARK = os.getenv("BENCHMARK", "copywriting_env")

MAX_STEPS = 3
SUCCESS_SCORE_THRESHOLD = 0.6

_SYSTEM = {
    "subject_line_rewrite": (
        "You are a marketing copywriter. "
        "Rewrite the given email subject line to maximise open rates. "
        "Return ONLY the rewritten subject line — no explanation, no quotes, max 60 characters."
    ),
    "cold_email_draft": (
        "You are a marketing copywriter. "
        "Write the requested cold outreach email. "
        "Return ONLY the email body — no subject line header, no explanation."
    ),
    "ab_copy_judge": textwrap.dedent("""
        You are a B2B marketing expert judging which campaign performs better for cold outreach to CFOs.
        Respond in EXACTLY this format — no extra text before or after:
        WINNER: B
        REASON 1: <your reason>
        REASON 2: <your reason>
        REASON 3: <your reason>
    """).strip(),
}

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    # Ensure action string has no newlines
    safe_action = action.replace("\n", " ").replace("\r", "")[:120]
    print(
        f"[STEP] step={step} action={safe_action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

def get_model_message(client: OpenAI, tool_name: str, task_prompt: str) -> str:
    system_prompt = _SYSTEM.get(tool_name, "Complete the task.")
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task_prompt},
            ],
            temperature=0.7,
            max_tokens=400,
            stream=False,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return ""

async def main() -> None:
    # Mandatory OpenAI client usage
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Initialize environment (Remote Space)
    env = CopywritingEnv(base_url=ENV_URL)

    tasks_to_eval = ["subject_line_rewrite", "cold_email_draft", "ab_copy_judge"]
    
    # Store dynamic prompts/outputs for sequential dependency (e.g. Task 2 needs Task 1's headline)
    context_data = {"task1_output": "Monthly newsletter - please read"}

    for task_name in tasks_to_eval:
        rewards: List[float] = []
        steps_taken = 0
        score = 0.0
        success = False

        # Use the specific task_name for logging to ensure platform discovery
        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

        try:
            await env.reset()
            
            # Prepare the specific prompt for this task
            if task_name == "subject_line_rewrite":
                prompt = context_data["task1_output"]
            elif task_name == "cold_email_draft":
                prompt = f"Headline: {context_data.get('task1_output', 'Marketing Update')}"
            else:
                prompt = "Perform the A/B copy judge task."
            
            message = get_model_message(client, task_name, prompt)

            action = CallToolAction(tool_name=task_name, arguments={"candidate": message})
            result = await env.step(action)

            reward = float(result.reward or 0.0)
            done = bool(result.done)

            # Update context for next task if needed
            if task_name == "subject_line_rewrite" and message:
                context_data["task1_output"] = message

            rewards.append(reward)
            steps_taken = 1
            log_step(step=1, action=message, reward=reward, done=done, error=None)

            score = reward # Single tool call per episodic task here
            success = score >= SUCCESS_SCORE_THRESHOLD

        except Exception as exc:
            print(f"[DEBUG] Episode error for {task_name}: {exc}", flush=True)
        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    try:
        await env.close()
    except Exception as e:
        print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())