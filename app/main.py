from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from env.core import SupplyChainEnv
from env.schemas import SupplyChainAction

# Initialize the FastAPI application
app = FastAPI(
    title="OpenEnv: Supply Chain Optimizer",
    description="A multi-echelon inventory management environment for Agentic RL.",
    version="1.0.0"
)

# Instantiate our global environment instance
active_env = SupplyChainEnv()

# Helper model so the user can specify which task to load
class ResetRequest(BaseModel):
    task_id: str = "task_01_easy"

@app.get("/")
def health_check():
    """Automated ping for Hugging Face Spaces. Must return 200."""
    return {"status": "healthy", "message": "OpenEnv is running!"}

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
    """
    Hackathon requirement: Trigger the baseline script.
    Note: For a production HF Space, you'd run this as a background task.
    """
    return {"message": "Baseline endpoint active. To run full eval, execute baseline.py locally."}