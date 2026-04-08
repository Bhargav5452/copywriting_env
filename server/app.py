import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from openenv.core.env_server.web_interface import create_app

# Ensure environment module is discoverable
sys.path.append(str(Path(__file__).parent))

try:
    from environment import CopywritingEnvironment
except ImportError:
    from .environment import CopywritingEnvironment

app: FastAPI = create_app(lambda: CopywritingEnvironment())

if __name__ == "__main__":
    # Primary port for HF Spaces is 7860
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)