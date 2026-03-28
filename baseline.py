import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from env.core import SupplyChainEnv
from env.schemas import SupplyChainAction

# Load environment variables
load_dotenv()

# =================================================================
# 🔵 AI CLIENT CONFIGURATION(Global so the UI can use it too)
# =================================================================
# Uncomment the Groq block below when you are ready to switch from Ollama
"""
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY") 
)
model_name = "" 
"""

# Currently set to local Ollama testing
client = OpenAI(
    base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    api_key="ollama" 
)
model_name = "llama3.2:3b"
# =================================================================


def get_agent_action(obs):
    """
    Takes the current environment observation and runs the Multi-Agent pipeline.
    Returns: (SupplyChainAction, reasoning_string)
    """
    # ---------------------------------------------------------
    # AGENT 1: THE ANALYST (Reasoning & Math)
    # ---------------------------------------------------------
    analyst_prompt = (
        "You are the Lead Supply Chain Analyst. Your goal is to maximize profit and avoid bankruptcy.\n"
        "Laptops cost $800 each. Standard shipping takes 3 days.\n"
        "CRITICAL STRATEGY:\n"
        "1. NEVER let the cash balance drop below $8,000.\n"
        "2. If 'CURRENT STOCK' is less than 20 AND you have at least $20,000 in cash, order exactly 20 laptops.\n"
        "3. If 'CURRENT STOCK' is 20 or more, OR cash is low, order 0 laptops to let stock arrive.\n"
        "Analyze the state and write a 1-sentence plan stating exactly how many laptops to order today."
    )
    
    analyst_context = (
        f"MARKET SIGNAL: {obs.market_trend_signal}\n"
        f"CASH BALANCE: ${obs.cash_balance}\n"
        f"CURRENT STOCK: {obs.warehouse_status[0].current_stock}\n"
        "Write your plan:"
    )

    try:
        analyst_response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": analyst_prompt},
                {"role": "user", "content": analyst_context}
            ],
            temperature=0.2 
        )
        strategic_plan = analyst_response.choices[0].message.content

    except Exception as e:
        strategic_plan = f"Analyst Error: {e}. Defaulting to 0."

    # ---------------------------------------------------------
    # AGENT 2: THE EXECUTOR (JSON Formatting)
    # ---------------------------------------------------------
    executor_prompt = (
        "You are a strict Data Entry Agent. Convert the Analyst's plan into the exact JSON format required.\n"
        "RULES:\n"
        "1. The 'product_id' MUST ALWAYS BE exactly \"SKU-LAPTOP\". Do not use any other string.\n"
        "2. The 'expedite_shipping' field MUST BE false.\n"
        "3. If the plan says do nothing or 0, output an empty orders array: {\"orders\": []}\n"
        "4. Output ONLY valid JSON. No markdown blocks.\n"
        f"SCHEMA REQUIRED: {SupplyChainAction.model_json_schema()}"
    )

    try:
        executor_response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": executor_prompt},
                {"role": "user", "content": f"ANALYST PLAN: {strategic_plan}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.0 
        )
        
        content = executor_response.choices[0].message.content
        action = SupplyChainAction.model_validate_json(content)
        
    except Exception as e:
        action = SupplyChainAction(orders=[])

    return action, strategic_plan


def run_baseline():
    """
    Runs the full CLI hackathon evaluation loop.
    """
    env = SupplyChainEnv()
    tasks = ["task_01_easy", "task_02_medium", "task_03_hard"]
    report_card = {}

    print("\n" + "═"*70)
    print("🚀 OPENENV MULTI-AGENT EVALUATION: START")
    print("═"*70)

    for task_id in tasks:
        print(f"\n📍 STARTING {task_id.upper()}")
        obs = env.reset(task_id)
        done = False

        while not done:
            # ---> MAGIC HAPPENS HERE <---
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
    run_baseline()