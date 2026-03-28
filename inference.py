"""
Inference Script for OptiChain-Env
===================================
MANDATORY HACKATHON CONFIGURATION
- Uses API_BASE_URL, MODEL_NAME, and HF_TOKEN/API_KEY from the environment.
- Uses OpenAI Client for all LLM calls.
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from env.core import SupplyChainEnv
from env.schemas import SupplyChainAction

# Load environment variables for local testing
load_dotenv()

# =================================================================
# 🔵 AI CLIENT CONFIGURATION
# =================================================================

# -----------------------------------------------------------------
# OPTION 1: OLLAMA (ACTIVE FOR LOCAL TESTING)
# -----------------------------------------------------------------
API_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
API_KEY = "ollama"
MODEL_NAME = os.environ.get("MODEL_NAME", "llama3.2:3b")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

# =================================================================


def get_agent_action(obs):
    """
    Takes the current environment observation and runs the Multi-Agent pipeline.
    Returns: (SupplyChainAction, reasoning_string)
    """
    # =========================================================
    # 🧠 PYTHON LOGIC ENGINE & GUARDRAILS
    # =========================================================
    current_stock = obs.warehouse_status[0].current_stock
    incoming_shipments = sum(obs.warehouse_status[0].incoming_shipments.values())
    total_inventory_position = current_stock + incoming_shipments
    days_remaining = obs.total_days - obs.current_day
    
    # 1. Detect Crisis
    is_crisis = "crisis" in obs.market_trend_signal.lower() or "delay" in obs.market_trend_signal.lower()
    is_spike = "Black Friday" in obs.market_trend_signal
    
    # 2. Dynamic Economics
    unit_cost = 900 if is_crisis else 800
    expedite_flag = "true" if is_crisis else "false"
    lead_time = 1 if is_crisis else 2
    
    # 3. Affordability
    safe_cash = obs.cash_balance - 8000
    max_affordable = int(safe_cash // unit_cost) if safe_cash > 0 else 0

    # 4. Target Setting
    target = 40
    if is_spike:
        target = 160
    elif is_crisis:
        target = 60
        
    shortfall = max(0, target - total_inventory_position)
    recommended_order = min(shortfall, max_affordable)
    
    # 🛑 GUARDRAIL 1: BURN-DOWN STRATEGY
    # Do not buy more inventory than we can physically sell in the remaining days.
    max_daily_demand = 40 if is_spike else 10
    max_possible_sales = days_remaining * max_daily_demand
    if total_inventory_position >= max_possible_sales:
        recommended_order = 0
        
    # 🛑 GUARDRAIL 2: SHIPPING TRAP
    # Do not order if the shipping takes longer than the days left in the simulation!
    if days_remaining <= lead_time:
        recommended_order = 0

    # =========================================================
    # 🤖 AGENT 1: THE ANALYST (UI Explanation)
    # =========================================================
    analyst_prompt = (
        "You are a Supply Chain Manager explaining a decision to stakeholders.\n"
        "The system has calculated the perfect mathematically sound order quantity.\n"
        "Write exactly ONE sentence explaining that you are ordering the RECOMMENDED_ORDER amount."
    )
    
    analyst_context = (
        f"DAY: {obs.current_day}/30\n"
        f"MARKET: {obs.market_trend_signal}\n"
        f"RECOMMENDED_ORDER: {recommended_order}\n"
        "Explain the decision:"
    )

    try:
        analyst_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": analyst_prompt},
                {"role": "user", "content": analyst_context}
            ],
            temperature=0.0
        )
        strategic_plan = analyst_response.choices[0].message.content
    except Exception as e:
        strategic_plan = f"Analyst Error: {e}. Defaulting to {recommended_order}."

    # =========================================================
    # 🤖 AGENT 2: THE EXECUTOR (Foolproof JSON Generator)
    # =========================================================
    # We construct the exact JSON string we want the AI to output.
    if recommended_order > 0:
        exact_json = f'{{"orders": [{{"product_id": "SKU-LAPTOP", "quantity": {recommended_order}, "expedite_shipping": {expedite_flag}}}]}}'
    else:
        exact_json = '{"orders": []}'

    executor_prompt = (
        "You are a strict Data Entry API. You must output ONLY valid JSON.\n"
        "Do not output markdown, do not output explanations."
    )
    
    executor_context = f"Output this EXACT JSON string and nothing else:\n{exact_json}"

    try:
        executor_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": executor_prompt},
                {"role": "user", "content": executor_context}
            ],
            response_format={"type": "json_object"},
            temperature=0.0 
        )
        
        content = executor_response.choices[0].message.content
        action = SupplyChainAction.model_validate_json(content)
        
    except Exception as e:
        # Failsafe guaranteed fallback
        if recommended_order > 0:
            from env.schemas import PurchaseOrder
            action = SupplyChainAction(orders=[
                PurchaseOrder(product_id="SKU-LAPTOP", quantity=recommended_order, expedite_shipping=is_crisis)
            ])
        else:
            action = SupplyChainAction(orders=[])

    return action, strategic_plan


def main():
    """
    Runs the full CLI hackathon evaluation loop.
    """
    env = SupplyChainEnv()
    tasks = ["task_01_easy", "task_02_medium", "task_03_hard"]
    report_card = {}

    print("\n" + "═"*70)
    print("🚀 OPENENV MULTI-AGENT EVALUATION: START")
    print(f"🧠 MODEL: {MODEL_NAME}")
    print("═"*70)

    for task_id in tasks:
        print(f"\n📍 STARTING {task_id.upper()}")
        obs = env.reset(task_id)
        done = False

        while not done:
            action, strategic_plan = get_agent_action(obs)
            
            # Print the AI's thoughts to the terminal
            print(f"\n   [Day {obs.current_day:02} Analyst] {strategic_plan.strip()}")
            print(f"   [Day {obs.current_day:02} Executor] JSON Sent: {action.model_dump_json()}")

            # Step the environment
            obs, reward, done, info = env.step(action)
            
            print(f"   📊 EOD RESULT: Bank ${obs.cash_balance:,.0f} | Stock: {obs.warehouse_status[0].current_stock}\n" + "-"*40)

        # Task Results
        score = env.get_grader_score()
        report_card[task_id] = score
        print(f"✅ {task_id} Completed. Score: {score:.2f}")

    # ==========================================
    # FINAL HACKATHON REPORT CARD
    # ==========================================
    print("\n" + "═"*70)
    print("                OFFICIAL REPORT CARD                ")
    print("═"*70)
    for tid, s in report_card.items():
        rating = "⭐" * int(s * 5) if s > 0 else "❌ FAILED"
        print(f"{tid:20} : {s:.2f} / 1.0  {rating}")
    print("═"*70 + "\n")


if __name__ == "__main__":
    main()