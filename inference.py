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
from dotenv import load_dotenv
from openai import OpenAI
from env.core import SupplyChainEnv
from env.schemas import SupplyChainAction, SupplyChainObservation

# Load environment variables for local testing
load_dotenv()

# =================================================================
# AI CLIENT CONFIGURATION
# =================================================================

# --- OPTION 1: OLLAMA (Active for Local Testing) ---
API_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
API_KEY      = "ollama"
MODEL_NAME   = os.environ.get("MODEL_NAME") or "llama3.2:3b"

# --- OPTION 2: GROQ CLOUD (Commented Out) ---
# API_BASE_URL = "https://api.groq.com/openai/v1"
# API_KEY      = os.environ.get("GROQ_API_KEY")
# MODEL_NAME   = "llama-3.3-70b-versatile"

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
# =================================================================


def get_agent_action(obs: SupplyChainObservation) -> tuple[SupplyChainAction, str]:
    """
    Multi-agent pipeline: Analyst reasons about the market, Executor formats the JSON.
    Returns: (SupplyChainAction, reasoning_string)
    """
    # =========================================================
    # 1. EXTRACT TELEMETRY FOR THE LLM
    # =========================================================
    wh = obs.warehouse_status[0]
    current_stock = wh.current_stock
    incoming_shipments = sum(wh.incoming_shipments.values())
    total_inventory_pos = current_stock + incoming_shipments
    days_remaining = obs.total_days - obs.current_day
    
    sales_yesterday = wh.sales_yesterday
    lost_yesterday = wh.lost_sales_yesterday

    # Determine dynamic costs to give the LLM accurate budget info
    is_crisis = "crisis" in obs.market_trend_signal.lower() or "delay" in obs.market_trend_signal.lower()
    std_cost = 800
    exp_cost = 900 if is_crisis else 160 # In standard mode, expedite is 800+100? Wait, core.py says cost += 100. So 900 always for expedite.
    exp_cost = 900 
    
    # Calculate affordability limits (keeping $8,000 safe buffer)
    safe_cash = obs.cash_balance - 8000
    max_afford_std = int(safe_cash // std_cost) if safe_cash > 0 else 0
    max_afford_exp = int(safe_cash // exp_cost) if safe_cash > 0 else 0

    # Format the pipeline so the LLM knows exactly when stock arrives
    pipeline_str = ", ".join([f"{qty} units in {days} days" for days, qty in wh.incoming_shipments.items() if qty > 0])
    if not pipeline_str:
        pipeline_str = "No incoming shipments."

    # =========================================================
    # 🤖 AGENT 1: THE ANALYST (Strategic Decision Maker)
    # =========================================================
    analyst_prompt = (
        "You are an elite AI Supply Chain Manager. Your goal is to maximize total profit over a 30-day period.\n\n"
        "=== UNIT ECONOMICS ===\n"
        "- Standard Order: $800 cost, takes 2 days (4 days during crisis).\n"
        "- Expedited Order: $900 cost, takes 1 day.\n"
        "- Revenue per sale: $1,200\n"
        "- Holding cost: $2 / unit / day\n"
        "- Stockout penalty: $100 / missed sale\n\n"
        "=== STRATEGY GUIDELINES ===\n"
        "1. Predict demand based on the Market Signal and yesterday's sales.\n"
        "2. Check your 'Inventory Position' (Stock + Incoming). Do you have enough to cover the lead time?\n"
        "3. If a spike is coming, pre-order heavily. If a crisis is active, use expedited shipping to avoid $100/day penalties.\n"
        "4. DO NOT order more units than the 'Max Affordable' limit.\n"
        "5. As you approach Day 30, burn down your stock to 0. Do not order stock that will arrive after Day 30.\n\n"
        "You must output a short paragraph of reasoning, followed EXACTLY by these two lines at the very end:\n"
        "ORDER_QUANTITY: [number]\n"
        "EXPEDITE: [true/false]"
    )
    
    analyst_context = (
        f"=== CURRENT STATUS ===\n"
        f"DAY: {obs.current_day} of {obs.total_days} ({days_remaining} days remaining)\n"
        f"MARKET SIGNAL: {obs.market_trend_signal}\n"
        f"YESTERDAY'S PERFORMANCE: Sold {sales_yesterday}, Missed {lost_yesterday} sales.\n"
        f"CASH BALANCE: ${obs.cash_balance:.2f}\n"
        f"CURRENT STOCK: {current_stock} units\n"
        f"PIPELINE: {pipeline_str}\n"
        f"TOTAL INVENTORY POSITION: {total_inventory_pos} units\n\n"
        f"=== BUDGET LIMITS ===\n"
        f"Max Affordable (Standard): {max_afford_std} units\n"
        f"Max Affordable (Expedited): {max_afford_exp} units\n\n"
        "Write your reasoning and final decision:"
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": analyst_prompt},
                {"role": "user",   "content": analyst_context},
            ],
            temperature=0.1, # Slight temperature for reasoning, but kept low for math stability
        )
        strategic_plan = resp.choices[0].message.content
    except Exception as exc:
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
        )
        action = SupplyChainAction.model_validate_json(resp.choices[0].message.content)
    except Exception:
        action = SupplyChainAction(orders=[])

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

            print("[STEP]", json.dumps({
                "task_id":   task_id,
                "day":       obs.current_day,
                "action":    action.model_dump(),
                "reasoning": strategic_plan.strip(),
            }))

            obs = env.step(action)

            print("[STEP]", json.dumps({
                "task_id": task_id,
                "day":     obs.current_day,
                "reward":  round(obs.reward, 2),
                "done":    obs.done,
                "cash":    round(obs.cash_balance, 2),
                "stock":   obs.warehouse_status[0].current_stock if obs.warehouse_status else 0,
            }))

        score = env.get_grader_score()
        report_card[task_id] = score
        print("[END]", json.dumps({"task_id": task_id, "score": round(score, 4)}))

    # Final summary
    print("[END]", json.dumps({"report_card": report_card}))


if __name__ == "__main__":
    main()