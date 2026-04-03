"""
OptiChain-Env: OpenEnv-compliant Supply Chain Optimization Environment
=======================================================================
Entry point for the FastAPI server.

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from env.core import SupplyChainEnv
from env.schemas import SupplyChainAction, SupplyChainObservation, SupplyChainState
from inference import get_agent_action

# ---------------------------------------------------------------------------
# Single shared environment instance (single-session, per OpenEnv spec)
# ---------------------------------------------------------------------------
env = SupplyChainEnv()

# ---------------------------------------------------------------------------
# App creation — use openenv create_app when available for spec compliance,
# fall back to plain FastAPI so the server still boots without the package.
# ---------------------------------------------------------------------------
try:
    from openenv.core.env_server import create_app
    app = create_app(
        SupplyChainEnv,
        SupplyChainAction,
        SupplyChainObservation,
        env_name="optichain-inventory-v1",
    )
except ImportError:
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


# ---------------------------------------------------------------------------
# OpenEnv-required endpoints
# POST /reset  — task_id selects Easy / Medium / Hard scenario
# POST /step   — advance simulation one day
# GET  /state  — current episode metadata (no side effects)
# GET  /grader — normalized score 0.0–1.0
# GET  /health — liveness probe for Hugging Face Spaces
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "task_01_easy"


@app.post("/reset", response_model=SupplyChainObservation)
def reset_env(req: ResetRequest):
    """Wipe state and load a specific task (easy / medium / hard)."""
    valid = {"task_01_easy", "task_02_medium", "task_03_hard"}
    if req.task_id not in valid:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{req.task_id}'. Valid: {sorted(valid)}")
    return env.reset(task_id=req.task_id)


class StepResponse(BaseModel):
    observation: SupplyChainObservation
    reward: float
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
        reward=obs.reward,
        done=obs.done,
        info={"current_profit": env.total_profit},
    )


@app.get("/state", response_model=SupplyChainState)
def get_state():
    """Return current episode metadata (step count, profit, grader score) without advancing time."""
    return env.state


class GraderResponse(BaseModel):
    score: float


@app.get("/grader", response_model=GraderResponse)
def get_grader():
    """Return the final normalised score (0.0–1.0) for the current episode."""
    return GraderResponse(score=env.get_grader_score())


@app.get("/health")
def health_check():
    """Liveness probe — must return 200 for Hugging Face Spaces automated pings."""
    return {"status": "healthy", "message": "OpenEnv OptiChain is running!"}


# ---------------------------------------------------------------------------
# Extra endpoints (task listing, UI demo bridge)
# ---------------------------------------------------------------------------

@app.get("/tasks")
def get_tasks():
    """List available tasks and the action JSON schema."""
    return {
        "tasks": [
            {"id": "task_01_easy",   "name": "Stable Store",         "difficulty": "easy"},
            {"id": "task_02_medium", "name": "Holiday Demand Spike",  "difficulty": "medium"},
            {"id": "task_03_hard",   "name": "Global Supply Shock",   "difficulty": "hard"},
        ],
        "action_schema": SupplyChainAction.model_json_schema(),
    }


@app.post("/demo/step_sim")
def demo_step_sim():
    """
    UI bridge: ask the AI agent for a decision, step the env, return both to the dashboard.
    """
    if not env.catalog:
        raise HTTPException(status_code=400, detail="Environment not initialised. Call POST /reset first.")

    current_obs = env._build_observation(reward=0.0, done=False)

    try:
        action, reasoning = get_agent_action(current_obs)
    except Exception as exc:
        action = SupplyChainAction(orders=[])
        reasoning = f"Agent error: {exc}. Defaulting to zero orders."

    obs = env.step(action)
    return {
        "observation": obs,
        "reward": obs.reward,
        "done": obs.done,
        "action_taken": action.model_dump(),
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
