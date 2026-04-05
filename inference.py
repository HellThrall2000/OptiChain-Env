"""
Inference Script for OptiChain-Env
===================================
MANDATORY HACKATHON CONFIGURATION
- Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from environment.
- Uses OpenAI Python client for all LLM calls.
- Emits structured [START], [STEP], [END] logs for automated scoring.
"""

import os
import json
import logging
from dotenv import load_dotenv
from openai import OpenAI
from env.core import SupplyChainEnv, EXPEDITE_SURCHARGE
from env.schemas import SupplyChainAction, SupplyChainObservation, PurchaseOrder

logger = logging.getLogger(__name__)

# Load environment variables for local testing
load_dotenv()

# =================================================================
# AI CLIENT CONFIGURATION (hackathon-required env var names)
# Uses `or` so empty strings in .env still fall back to defaults.
# Local Ollama: set API_BASE_URL=http://localhost:11434/v1 in .env
# Cloud (Groq):  set API_BASE_URL=https://api.groq.com/openai/v1
# =================================================================
API_BASE_URL = os.environ.get("API_BASE_URL") or "http://localhost:11434/v1"
API_KEY      = (os.environ.get("HF_TOKEN")
                or os.environ.get("API_KEY")
                #or os.environ.get("GROQ_API_KEY")
                or "ollama")
MODEL_NAME   = os.environ.get("MODEL_NAME") or "llama3.1:8b"

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
# =================================================================


def get_agent_action(obs: SupplyChainObservation) -> tuple[SupplyChainAction, str]:
    """
    Multi-agent pipeline: Analyst reasons about the market, Executor formats the JSON.
    Returns: (SupplyChainAction, reasoning_string)
    """
    # =========================================================
    # 1. EXTRACT RAW TELEMETRY FOR THE LLM (No Python Math)
    # =========================================================
    wh = obs.warehouse_status[0]
    current_stock = wh.current_stock
    incoming_shipments = sum(wh.incoming_shipments.values())
    total_inventory_pos = current_stock + incoming_shipments
    days_remaining = obs.total_days - obs.current_day
    
    sales_yesterday = wh.sales_yesterday
    lost_yesterday = wh.lost_sales_yesterday

    # Format the pipeline so the LLM knows exactly when stock arrives
    pipeline_str = ", ".join([f"{qty} units in {days} days" for days, qty in wh.incoming_shipments.items() if qty > 0])
    if not pipeline_str:
        pipeline_str = "No incoming shipments."

    # =========================================================
    # 🤖 AGENT 1: THE ANALYST (Strategic Decision Maker)
    # =========================================================
    analyst_prompt = (
        "You are an Elite Supply Chain Optimizer. Your performance is graded on 'Inventory Efficiency' (Newsvendor Logic).\n\n"
        
        "=== THE GOLDEN RULES FOR A 1.0 SCORE ===\n"
        "1. OVERAGE IS FAILURE: Ending any day with unsold stock kills your score. Aim for JIT (Just-In-Time) delivery.\n"
        "2. UNDERAGE IS FAILURE: Missing a customer sale kills your score. Maintain a minimal safety buffer.\n"
        "3. PIPELINE MATH: Your 'Total Inventory Position' = Current Stock + All units in Pipeline.\n"
        "4. TARGET FORMULA: Aim for a Total Inventory Position = (Predicted Daily Demand) * (Lead Time + 1).\n\n"

        "=== UNIT ECONOMICS & LEAD TIMES ===\n"
        "- Standard: $800 cost | 2-day lead time (4 days during crisis).\n"
        "- Expedited: $900 cost | 1-day lead time.\n"
        "- Holding Cost: $2/unit/day | Stockout Penalty: $100/unit.\n\n"

        "=== STRATEGIC MANDATES ===\n"
        "- PREDICTIVE BUFFER: Analyze 'Yesterday's Performance'. If demand was 12, assume today is 12. Add a +2 unit safety buffer only.\n"
        "- BUDGET CHECK: You MUST multiply (Order Quantity * Unit Cost). This result MUST be less than your current Cash Balance.\n"
        "- CRISIS ADAPTATION: During a shipping crisis (4-day delay), use Expedited (1-day) to stay lean and responsive.\n"
        "- HORIZON AWARENESS: The simulation ends on Day 30. Any stock arriving after Day 30 is a total financial loss and results in a 0.0 efficiency score. "
        "Calculate 'Days Remaining' vs 'Lead Time' to decide when to stop ordering. Your goal is to have EXACTLY zero stock on Day 30.\n\n"

        "=== REQUIRED OUTPUT FORMAT ===\n"
        "Begin with a 'Step-by-Step Math' paragraph: Calculate Predicted Demand, Current Inventory Position, and identify if an order will arrive before Day 30.\n"
        "End your response with these EXACT tags:\n"
        "ORDER_QUANTITY: [number]\n"
        "EXPEDITE: [true/false]"
    )
    
    # Build order feedback line so the LLM knows if its last order was rejected
    if obs.last_order_rejected > 0:
        order_feedback = (
            f"LAST ORDER: REJECTED {obs.last_order_rejected} units (insufficient funds). "
            f"Only {obs.last_order_accepted} units were accepted. Reduce your order size!"
        )
    elif obs.last_order_accepted > 0:
        order_feedback = f"LAST ORDER: Accepted {obs.last_order_accepted} units."
    else:
        order_feedback = "LAST ORDER: No order placed."

    analyst_context = (
        f"=== CURRENT STATUS ===\n"
        f"DAY: {obs.current_day} of {obs.total_days} ({days_remaining} days remaining)\n"
        f"MARKET SIGNAL: {obs.market_trend_signal}\n"
        f"YESTERDAY'S PERFORMANCE: Sold {sales_yesterday}, Missed {lost_yesterday} sales.\n"
        f"{order_feedback}\n"
        f"CASH BALANCE: ${obs.cash_balance:.2f}\n"
        f"CURRENT STOCK: {current_stock} units\n"
        f"PIPELINE: {pipeline_str}\n"
        f"TOTAL INVENTORY POSITION: {total_inventory_pos} units\n\n"
        "Write your reasoning and final decision:"
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": analyst_prompt},
                {"role": "user",   "content": analyst_context},
            ],
            temperature=0.1,  # Slight temperature allows predictive flexibility
            timeout=30,
        )
        strategic_plan = resp.choices[0].message.content
    except Exception as exc:
        logger.error("Analyst agent failed: %s", exc, exc_info=True)
        strategic_plan = f"Analyst unavailable ({exc}).\nORDER_QUANTITY: 0\nEXPEDITE: false"

    # =========================================================
    # 🤖 AGENT 2: THE EXECUTOR (Strict JSON Formatter)
    # =========================================================
    executor_prompt = (
        "You are a strict Data Parsing API. Read the Analyst's plan, locate the 'ORDER_QUANTITY' and 'EXPEDITE' values, "
        "and output ONLY valid JSON using this exact schema:\n"
        "{\n"
        '  "orders": [\n'
        '    {"product_id": "SKU-LAPTOP", "quantity": <INT>, "expedite_shipping": <BOOL>}\n'
        "  ]\n"
        "}\n"
        "If ORDER_QUANTITY is 0, output: {\"orders\": []}\n"
        "Do not output markdown blocks or any other text."
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": executor_prompt},
                {"role": "user",   "content": f"ANALYST PLAN:\n{strategic_plan}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            timeout=30,
        )
        action = SupplyChainAction.model_validate_json(resp.choices[0].message.content)
    except Exception as exc:
        logger.error("Executor agent failed: %s", exc, exc_info=True)
        action = SupplyChainAction(orders=[])

    # =================================================================
    # PYTHON GUARDRAILS — clip LLM output to what's actually affordable
    # Prevents ghost orders and logic-tag dissonance from mattering.
    # =================================================================
    clipped_orders = []
    remaining_cash = obs.cash_balance
    for order in action.orders:
        # SKU-LAPTOP order cost is $800; expedited adds EXPEDITE_SURCHARGE per unit
        unit_cost = (800 + EXPEDITE_SURCHARGE) if order.expedite_shipping else 800
        max_affordable = int(remaining_cash // unit_cost) if remaining_cash > 0 else 0
        qty = min(order.quantity, max_affordable)
        if qty > 0:
            remaining_cash -= qty * unit_cost
            clipped_orders.append(PurchaseOrder(
                product_id=order.product_id,
                quantity=qty,
                expedite_shipping=order.expedite_shipping,
            ))
    action = SupplyChainAction(orders=clipped_orders)

    return action, strategic_plan


def main():
    """
    Full CLI evaluation loop.
    Emits [START], [STEP], [END] structured logs required by the hackathon scorer.
    """
    env   = SupplyChainEnv()
    tasks = ["task_01_easy", "task_02_medium", "task_03_hard"]
    report_card: dict[str, float] = {}

    for task_id in tasks:
        print("[START]", json.dumps({"task_id": task_id, "model": MODEL_NAME}))

        obs  = env.reset(task_id=task_id)

        while not obs.done:
            action, strategic_plan = get_agent_action(obs)

            requested = sum(o.quantity for o in action.orders)

            obs = env.step(action)

            wh = obs.warehouse_status[0] if obs.warehouse_status else None
            print("[STEP]", json.dumps({
                "task_id":   task_id,
                "day":       obs.current_day - 1,
                "ordered":   requested,
                "accepted":  env.last_accepted_qty,
                "rejected":  env.last_rejected_qty,
                "sold":      wh.sales_yesterday if wh else 0,
                "missed":    wh.lost_sales_yesterday if wh else 0,
                "stock":     wh.current_stock if wh else 0,
                "cash":      round(obs.cash_balance, 2),
                "reward":    round(obs.reward, 2),
                "done":      obs.done,
                "reasoning": strategic_plan.strip(),
            }))

        score = env.get_grader_score()
        report_card[task_id] = score
        print("[END]", json.dumps({"task_id": task_id, "score": round(score, 4)}))

    # Final summary
    print("[END]", json.dumps({"report_card": report_card}))


if __name__ == "__main__":
    main()