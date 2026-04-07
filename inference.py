import asyncio
import os
import textwrap
from typing import Optional, Any

from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

try:
    from client import CopywritingEnv
    from models import CallToolAction
except ImportError:
    from src.envs.copywriting_env.client import CopywritingEnv
    from src.envs.copywriting_env.models import CallToolAction

# API Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

TASK_NAME = "copywriting"
BENCHMARK = "copywriting_env"
MAX_STEPS = 3
SUCCESS_SCORE_THRESHOLD = 0.6

# Per-tool system prompts
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
    safe_action = action.replace("\n", " ")[:120]
    print(
        f"[STEP] step={step} action={safe_action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def ask_llm(provider: str, client: Any, tool_name: str, task_prompt: str) -> str:
    """Unified helper to call either OpenAI/HF or Google Gemini."""
    system_prompt = _SYSTEM[tool_name]
    try:
        if provider == "gemini":
            # client is a GenerativeModel instance
            response = client.generate_content(
                f"{system_prompt}\n\nTask: {task_prompt}"
            )
            return response.text.strip()
        else:
            # client is an OpenAI instance
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": task_prompt},
                ],
                temperature=0.7,
                max_tokens=400,
                stream=False,
            )
            return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] {provider} call failed for {tool_name}: {exc}", flush=True)
        return ""


async def main() -> None:
    # 1. Determine Provider
    provider = "openai"
    client = None
    
    if GEMINI_API_KEY:
        print(f"[INFO] Using Gemini API (Model: {GEMINI_MODEL})", flush=True)
        genai.configure(api_key=GEMINI_API_KEY)
        client = genai.GenerativeModel(GEMINI_MODEL)
        provider = "gemini"
    elif OPENAI_API_KEY or HF_TOKEN:
        print(f"[INFO] Using OpenAI/HF API (Model: {MODEL_NAME})", flush=True)
        client = OpenAI(base_url=API_BASE_URL, api_key=OPENAI_API_KEY or HF_TOKEN)
        provider = "openai"
    else:
        print("[ERROR] No API keys found (GEMINI_API_KEY, OPENAI_API_KEY, or HF_TOKEN)", flush=True)
        return

    env = CopywritingEnv(base_url=ENV_URL)
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    tools = ["subject_line_rewrite", "cold_email_draft", "ab_copy_judge"]
    task_prompts: dict[str, str] = {}

    log_start(task=TASK_NAME, env=BENCHMARK, model="Gemini" if provider == "gemini" else MODEL_NAME)

    try:
        await env.reset()

        for step, tool_name in enumerate(tools, start=1):
            prompt = task_prompts.get(tool_name, f"Complete the {tool_name} task.")
            candidate = ask_llm(provider, client, tool_name, prompt)

            action = CallToolAction(tool_name=tool_name, arguments={"candidate": candidate})
            result = await env.step(action)

            reward = float(result.reward or 0.0)
            done = bool(result.done)

            # Extract prompt for next tool if available
            if hasattr(result, "result") and result.result and hasattr(result.result, "data"):
                data = result.result.data or {}
                if isinstance(data, dict) and "prompt" in data:
                    # Update next tool prompts from the current task's response if any
                    pass 

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=candidate, reward=reward, done=done, error=None)

            if done:
                break

        score = sum(rewards) / MAX_STEPS
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        try:
            await env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close() error: {exc}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())