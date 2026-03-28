import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from env.core import SupplyChainEnv
from env.schemas import SupplyChainAction

# We import the AI logic from your baseline script
# (We will update baseline.py to export this function next)
from baseline import get_agent_action

# Initialize the FastAPI application
app = FastAPI(
    title="OpenEnv: Supply Chain Optimizer",
    description="A multi-echelon inventory management environment for Agentic RL.",
    version="1.0.0"
)

# --- 1. CORS MIDDLEWARE ---
# Required for Hugging Face Spaces to allow the UI to talk to the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate our global environment instance
active_env = SupplyChainEnv()

# --- 2. FRONTEND UI ROUTING ---
# Ensure the "static" directory exists in your root folder!
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_dashboard():
    """Serves the OptiChain HTML dashboard instantly on the root URL."""
    # Ensure your HTML file is named index.html and is inside the static/ folder
    return FileResponse("static/index.html")

@app.get("/health")
def health_check():
    """Automated ping for Hugging Face Spaces. Must return 200."""
    return {"status": "healthy", "message": "OpenEnv is running!"}


# --- 3. UI DEMO BRIDGE ---
@app.post("/demo/step_sim")
def demo_step_sim():
    """
    Bridge endpoint for the frontend. 
    It asks the AI Agent for a decision, steps the env, and returns both to the UI.
    """
    obs = active_env.state()
    
    # Ask the LLM (via baseline.py) what to do
    try:
        action, reasoning = get_agent_action(obs)
    except Exception as e:
        # Fallback if the API fails
        action = SupplyChainAction(orders=[])
        reasoning = f"API Error: {str(e)}. Defaulting to 0 orders."

    # Execute the action in the environment
    next_obs, reward, done, info = active_env.step(action)
    
    return {
        "observation": next_obs,
        "reward": reward,
        "done": done,
        "action_taken": action.model_dump(),
        "reasoning": reasoning
    }


# --- 4.OFFICIAL OPENENV ENDPOINTS---
class ResetRequest(BaseModel):
    task_id: str = "task_01_easy"

@app.post("/reset")
def reset_env(req: ResetRequest = None):
    """Wipes the state and loads a specific task (easy, medium, hard)."""
    task_id = req.task_id if req else "task_01_easy"
    try:
        obs = active_env.reset(task_id=task_id)
        return {"observation": obs}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
def step_env(action: SupplyChainAction):
    """Processes the agent's action (PurchaseOrders) and advances time by 1 day."""
    obs, reward, done, info = active_env.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
def get_state():
    """Returns the current state without advancing time."""
    return {"observation": active_env.state()}

@app.get("/tasks")
def get_tasks():
    """Returns available tasks and the action schema for the agent to read."""
    tasks = [
        {"id": "task_01_easy", "name": "Stable Store", "difficulty": "easy"},
        {"id": "task_02_medium", "name": "Holiday Demand Spike", "difficulty": "medium"},
        {"id": "task_03_hard", "name": "Global Supply Shock", "difficulty": "hard"}
    ]
    return {
        "tasks": tasks,
        "action_schema": SupplyChainAction.model_json_schema()
    }

@app.get("/grader")
def get_grader():
    """Returns the final normalized score (0.0 to 1.0) for the judging criteria."""
    return {"score": active_env.get_grader_score()}

@app.post("/baseline")
def trigger_baseline():
    """Hackathon requirement: Trigger the baseline script."""
    return {"message": "Baseline endpoint active. To run full eval, execute baseline.py locally."}