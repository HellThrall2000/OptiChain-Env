from env.core import SupplyChainEnv
from env.schemas import SupplyChainAction, PurchaseOrder

def run_test():
    # 1. Initialize the environment
    env = SupplyChainEnv()

    # 2. Reset to start the episode (Loading Task 1)
    print("=== INITIAL STATE (DAY 1) ===")
    initial_obs = env.reset("task_01_easy")
    # model_dump_json() is a Pydantic method that makes the output look pretty
    print(initial_obs.model_dump_json(indent=2))

    # 3. Create a dummy action: Let's order 15 laptops standard delivery
    dummy_action = SupplyChainAction(
        orders=[
            PurchaseOrder(
                product_id="SKU-LAPTOP",
                quantity=15,
                expedite_shipping=False
            )
        ]
    )

    # 4. Take a step into Day 2
    print("\n=== AFTER STEPPING (DAY 2) ===")
    obs, reward, done, info = env.step(dummy_action)
    
    print(f"Daily Net Cash Flow (Reward): ${reward}")
    print(f"Is Episode Done?: {done}")
    print("\nNew State:")
    print(obs.model_dump_json(indent=2))

if __name__ == "__main__":
    run_test()