"""
OptiChain-Env: OpenEnv-compliant Supply Chain Optimization Environment
=======================================================================
Entry point for the FastAPI server.

Required by:
  - openenv validate spec  (checks server/app.py exists)
  - openenv.yaml           (app: server.app:app)
  - Dockerfile             (CMD uvicorn server.app:app)

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

import logging
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

from env.core import SupplyChainEnv
from env.schemas import (
    SupplyChainAction,
    SupplyChainObservation,
    SupplyChainState,
    SupplyChainReward,
)
from inference import get_agent_action

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Single shared environment instance (single-session, per OpenEnv spec)
# ---------------------------------------------------------------------------
env = SupplyChainEnv()

# ---------------------------------------------------------------------------
# FastAPI app — we build our own rather than using create_app() because:
#   1. create_app() instantiates a SECOND env internally (out of sync)
#   2. create_app() registers /reset, /step, /state first — our custom
#      versions with task_id support would be shadowed (FastAPI first-match)
#   3. We still inherit from the OpenEnv base classes (Environment, Action,
#      Observation, State) — the spec validator checks endpoint behaviour,
#      not whether create_app() was called.
# ---------------------------------------------------------------------------
app = FastAPI(
    title="OpenEnv: OptiChain Supply Chain Optimizer",
    description="A supply chain inventory management environment for agentic RL.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Static dashboard
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_dashboard():
    return FileResponse("static/index.html")


# ===================================================================
# OpenEnv-required endpoints
# POST /reset  — task_id selects Easy / Medium / Hard scenario
# POST /step   — advance simulation one day
# GET  /state  — current episode metadata (no side effects)
# GET  /grader — normalized score 0.0–1.0
# GET  /health — liveness probe for Hugging Face Spaces
# ===================================================================

class ResetRequest(BaseModel):
    task_id: str = "task_01_easy"
    seed: Optional[int] = None


@app.post("/reset", response_model=SupplyChainObservation)
def reset_env(req: Optional[ResetRequest] = Body(default=None)):
    """Wipe state and load a specific task (easy / medium / hard).

    The body is optional — the OpenEnv validator POSTs with no body, so we
    fall back to the default task_01_easy when req is None.
    """
    if req is None:
        req = ResetRequest()
    try:
        return env.reset(task_id=req.task_id, seed=req.seed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


class StepResponse(BaseModel):
    observation: SupplyChainObservation
    reward: SupplyChainReward
    done: bool
    info: dict


@app.post("/step", response_model=StepResponse)
def step_env(action: SupplyChainAction):
    """Execute the agent's purchase orders and advance the simulation by one day."""
    if not env.catalog:
        raise HTTPException(status_code=400, detail="Environment not initialised. Call POST /reset first.")
    obs = env.step(action)
    return StepResponse(
        observation=obs,
        reward=SupplyChainReward(
            step_reward=obs.reward,
            total_reward=env.total_reward,
            grader_score=env.get_grader_score(),
        ),
        done=obs.done,
        info={
            "accepted": env.last_accepted_qty,
            "rejected": env.last_rejected_qty,
        },
    )


@app.get("/state", response_model=SupplyChainState)
def get_state():
    """Return current episode metadata (step count, reward, grader score) without advancing time."""
    return env.state


class GraderResponse(BaseModel):
    score: float


@app.get("/grader", response_model=GraderResponse)
def get_grader():
    """Return the final normalised score (0.0-1.0) for the current episode."""
    return GraderResponse(score=env.get_grader_score())


@app.get("/health")
def health_check():
    """Liveness probe — must return 200 for Hugging Face Spaces automated pings."""
    return {"status": "healthy"}


@app.get("/metadata")
def get_metadata():
    """OpenEnv-required: returns environment name and description."""
    return {
        "name": "optichain-inventory-v1",
        "description": (
            "AI-driven supply chain inventory optimization. "
            "The agent acts as a Supply Chain Manager, placing daily purchase orders "
            "to maximise profit across a 30-day simulation."
        ),
        "version": "1.0.0",
        "tasks": [t["id"] for t in [
            {"id": "task_01_easy"},
            {"id": "task_02_medium"},
            {"id": "task_03_hard"},
        ]],
    }


@app.post("/mcp")
def mcp_endpoint(request: dict):
    """OpenEnv-required: minimal JSON-RPC 2.0 handler for MCP compliance."""
    method = request.get("method", "")
    req_id = request.get("id", 1)

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "optichain-inventory-v1", "version": "1.0.0"},
            },
        }

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "reset",
                        "description": "Reset the environment to a specific task.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"task_id": {"type": "string"}},
                            "required": ["task_id"],
                        },
                    },
                    {
                        "name": "step",
                        "description": "Advance the simulation one day with purchase orders.",
                        "inputSchema": SupplyChainAction.model_json_schema(),
                    },
                ]
            },
        }

    # Unknown method — return JSON-RPC error
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


# ===================================================================
# Extra endpoints (task listing, schema, UI demo bridge)
# ===================================================================

@app.get("/tasks")
def get_tasks():
    """List available tasks and the action JSON schema."""
    return {
        "tasks": [
            {"id": "task_01_easy",   "name": "Stable Demand",         "difficulty": "easy"},
            {"id": "task_02_medium", "name": "Holiday Demand Spike",   "difficulty": "medium"},
            {"id": "task_03_hard",   "name": "Supply Chain Crisis",    "difficulty": "hard"},
        ],
        "action_schema": SupplyChainAction.model_json_schema(),
    }


@app.get("/schema")
def get_schema():
    """Return observation/action/state JSON schemas for agent introspection."""
    return {
        "observation": SupplyChainObservation.model_json_schema(),
        "action": SupplyChainAction.model_json_schema(),
        "reward": SupplyChainReward.model_json_schema(),
        "state": SupplyChainState.model_json_schema(),
    }


@app.post("/demo/step_sim")
def demo_step_sim():
    """UI bridge: ask the AI agent for a decision, step the env, return both to the dashboard."""
    if not env.catalog:
        raise HTTPException(status_code=400, detail="Environment not initialised. Call POST /reset first.")

    current_obs = env._build_observation(reward=0.0, done=False)

    try:
        action, reasoning = get_agent_action(current_obs)
    except Exception as exc:
        logger.error("Agent error in demo/step_sim: %s", exc, exc_info=True)
        action = SupplyChainAction(orders=[])
        reasoning = f"Agent error: {exc}. Defaulting to zero orders."

    obs = env.step(action)
    return {
        "observation": obs.model_dump(),
        # reward is a plain float so the dashboard JS can use it directly
        # (pnl chart, appendLog, etc.). Full breakdown is in reward_detail.
        "reward": obs.reward,
        "reward_detail": SupplyChainReward(
            step_reward=obs.reward,
            total_reward=env.total_reward,
            grader_score=env.get_grader_score(),
        ).model_dump(),
        "done": obs.done,
        "action_taken": action.model_dump(),
        "accepted": env.last_accepted_qty,
        "rejected": env.last_rejected_qty,
        "reasoning": reasoning,
    }


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------
def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
