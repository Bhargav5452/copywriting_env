import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.types import Action, Observation

sys.path.append(str(Path(__file__).parent))

try:
    from environment import CopywritingEnvironment
except ImportError:
    from .environment import CopywritingEnvironment

# Initialize FastAPI with required framework classes
app: FastAPI = create_app(
    env=lambda: CopywritingEnvironment(),
    action_cls=Action,
    observation_cls=Observation,
    env_name="copywriting"
)

def main():
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()