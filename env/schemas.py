from pydantic import BaseModel, Field
from typing import List, Dict

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

class SupplyChainObservation(BaseModel):
    current_day: int
    total_days: int
    cash_balance: float
    warehouse_status: List[ProductStatus]
    market_trend_signal: str            

# --- ACTION SPACE ---
class PurchaseOrder(BaseModel):
    product_id: str
    quantity: int = Field(..., ge=0, description="Amount to order")
    expedite_shipping: bool = Field(default=False, description="True for 1-day delivery, False for 3-day")

class SupplyChainAction(BaseModel):
    orders: List[PurchaseOrder]