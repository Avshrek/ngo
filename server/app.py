"""
FastAPI application for the N-GO Environment.

This custom app maintains a SINGLE persistent environment instance so that
state (task_name, config, done) persists across HTTP /reset and /step calls.

The OpenEnv `create_app` factory creates a new env per request (designed for
WebSocket/MCP sessions), which breaks our stateful REST-based inference.

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

import json
import traceback
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from .ngo_environment import NGOEnvironment
except ImportError:
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from server.ngo_environment import NGOEnvironment


# ---------------------------------------------------------------------------
# Persistent environment instance
# ---------------------------------------------------------------------------
_env: Optional[NGOEnvironment] = None


def _get_env() -> NGOEnvironment:
    """Get or create the singleton environment instance."""
    global _env
    if _env is None:
        _env = NGOEnvironment()
    return _env


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_name: str = "cost_hemorrhage"
    seed: Optional[int] = None
    episode_id: Optional[str] = None


class ActionPayload(BaseModel):
    tool_name: str
    arguments: Dict[str, Any] = {}


class StepRequest(BaseModel):
    action: ActionPayload


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="N-GO: Neural-Gateway Orchestrator",
    description="Self-Defending LLM Router — OpenEnv Environment",
    version="1.0.0",
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "environment": "ngo"}


@app.get("/metadata")
async def metadata():
    """Environment metadata."""
    return {
        "env_name": "ngo",
        "description": "Neural-Gateway Orchestrator — Self-Defending LLM Router",
        "tasks": ["cost_hemorrhage", "pii_leak", "jailbreak_cascade"],
        "version": "1.0.0",
    }


@app.post("/reset")
async def reset(request: ResetRequest):
    """Reset environment for a new episode."""
    env = _get_env()
    try:
        observation = env.reset(
            seed=request.seed,
            episode_id=request.episode_id,
            task_name=request.task_name,
        )
        return _serialize_observation(observation)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
async def step(request: StepRequest):
    """Execute a tool action and return observation."""
    env = _get_env()
    try:
        action = CallToolAction(
            tool_name=request.action.tool_name,
            arguments=request.action.arguments,
        )
        observation = env.step(action)
        return _serialize_observation(observation)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def state():
    """Get current environment state."""
    env = _get_env()
    try:
        s = env.state
        return {
            "episode_id": s.episode_id,
            "step_count": s.step_count,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Serialization helper
# ---------------------------------------------------------------------------

def _serialize_observation(obs) -> Dict[str, Any]:
    """Convert an Observation object to a JSON-safe dict."""
    result: Dict[str, Any] = {
        "done": getattr(obs, "done", False),
        "reward": getattr(obs, "reward", None),
    }

    # Serialize the observation body (metadata, tool results, etc.)
    obs_data = {}
    metadata = getattr(obs, "metadata", None)
    if metadata is not None:
        obs_data = _make_serializable(metadata)

    # Check if there's a tool result in the observation
    tool_result = getattr(obs, "tool_result", None)
    if tool_result is not None:
        obs_data = _make_serializable(tool_result)

    # For CallToolObservation, extract the structured data
    if hasattr(obs, "tool_name"):
        obs_data["tool_name"] = obs.tool_name
    if hasattr(obs, "result"):
        obs_data["result"] = _make_serializable(obs.result)
    if hasattr(obs, "error"):
        obs_data["error"] = obs.error

    result["observation"] = obs_data
    return result


def _make_serializable(obj: Any) -> Any:
    """Recursively convert an object to JSON-serializable form."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return {k: _make_serializable(v) for k, v in obj.__dict__.items()}
    return str(obj)


def main():
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
