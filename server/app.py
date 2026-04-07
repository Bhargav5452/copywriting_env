import json
import os
import sys
from pathlib import Path
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

os.environ["ENABLE_WEB_INTERFACE"] = "true"

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.types import Observation

# Add the current directory to sys.path so sibling modules (models, environment) can be found
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

try:
    from models import CallToolAction, CopywritingObservation
    from environment import CopywritingEnvironment
except ImportError:
    # Fallback for different execution contexts
    try:
        from .models import CallToolAction, CopywritingObservation
        from .environment import CopywritingEnvironment
    except (ImportError, ValueError):
        from src.envs.copywriting_env.models import CallToolAction, CopywritingObservation
        from src.envs.copywriting_env.server.environment import CopywritingEnvironment


class ParseArgumentsMiddleware(BaseHTTPMiddleware):
    """Gradio sends tool arguments as a JSON string; this middleware parses them back into a dict."""

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.method == "POST" and "step" in request.url.path:
            try:
                raw_body = await request.body()
                if raw_body:
                    data = json.loads(raw_body)
                    if isinstance(data.get("arguments"), str):
                        data["arguments"] = json.loads(data["arguments"])
                        raw_body = json.dumps(data).encode("utf-8")

                    async def receive():
                        return {"type": "http.request", "body": raw_body, "more_body": False}

                    request._receive = receive
            except Exception as e:
                print(f"[Middleware] Failed to parse arguments: {e}")

        return await call_next(request)


app = create_app(
    CopywritingEnvironment,
    CallToolAction,
    CopywritingObservation,
    env_name="copywriting_env",
)

# middleware must be added after create_app
app.add_middleware(ParseArgumentsMiddleware)


@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "copywriting_env",
        "message": "Environment running! Use the Gradio UI or connect via API.",
    }


def main() -> None:
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()