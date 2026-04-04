from pydantic import BaseModel, Field
from typing import List, Dict
from openenv.core.env_server.interfaces import Action, Observation, State

# --- ACTION SPACE ---
class PurchaseOrder(BaseModel):
    product_id: str
    quantity: int = Field(..., ge=0, description="Amount to order")
    expedite_shipping: bool = Field(default=False, description="True for 1-day delivery, False for 2-day standard")


class SupplyChainAction(Action):
    orders: List[PurchaseOrder] = Field(
        default_factory=list,
        description="List of purchase orders to place this day"
    )


# --- OBSERVATION SPACE ---
class ProductStatus(BaseModel):
    product_id: str
    current_stock: int
    incoming_shipments: Dict[int, int]
    sales_yesterday: int
    lost_sales_yesterday: int
    holding_cost_per_unit: float
    stockout_penalty_per_unit: float
    margin_per_unit: float


class SupplyChainObservation(Observation):
    current_day: int
    total_days: int
    cash_balance: float
    warehouse_status: List[ProductStatus]
    market_trend_signal: str
    # OpenEnv convention: reward and done are part of the observation
    reward: float = Field(0.0, description="Reward received from the last step")
    done: bool = Field(False, description="Whether the episode has ended")


# --- REWARD (typed Pydantic model, required by hackathon spec) ---
class SupplyChainReward(BaseModel):
    step_reward: float = Field(0.0, description="Newsvendor reward for this day [0-1]")
    total_reward: float = Field(0.0, description="Cumulative sum of all step rewards so far")
    grader_score: float = Field(0.0, description="Normalized episode score: total_reward / 30, clamped [0-1]")


# --- STATE (OpenEnv episode metadata) ---
class SupplyChainState(State):
    # Inherits from State: episode_id: str, step_count: int
    current_task_id: str = ""
    total_reward: float = 0.0   # cumulative sum of per-step newsvendor rewards
    grader_score: float = 0.0   # total_reward / max_days, clamped to [0.0, 1.0]