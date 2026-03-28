from typing import Tuple, Any, Dict
from env.schemas import SupplyChainAction, SupplyChainObservation, ProductStatus

class SupplyChainEnv:
    def __init__(self):
        self.current_task_id = None
        self.max_days = 30
        self.current_day = 0
        self.cash_balance = 0.0
        self.inventory = {}
        self.shipment_pipeline = {}
        self.history = {"sales_yesterday": {}, "lost_sales_yesterday": {}}
        self.catalog = {}
        self.market_signal = ""
        
        # Tracking metrics for the Grader (0.0 to 1.0)
        self.total_profit = 0.0
        self.optimal_profit_baseline = 0.0

    def reset(self, task_id: str = "task_01_easy") -> SupplyChainObservation:
        self.current_task_id = task_id
        self.current_day = 1
        self.total_profit = 0.0
        
        # ==========================================
        # TASK 1: EASY (Stable Demand)
        # ==========================================
        if task_id == "task_01_easy":
            self.cash_balance = 30000.0  
            self.catalog = {
                # SELLING PRICE = 1200 (Cost 800 + Margin 400)
                "SKU-LAPTOP": {"margin": 400.0, "holding_cost": 2.0, "penalty": 100.0, "order_cost": 800.0}
            }
            self.inventory = {"SKU-LAPTOP": 50}
            self.shipment_pipeline = {"SKU-LAPTOP": {1: 0, 2: 0, 3: 0, 4: 0}}
            self.history = {"sales_yesterday": {"SKU-LAPTOP": 0}, "lost_sales_yesterday": {"SKU-LAPTOP": 0}}
            self.market_signal = "Demand is stable at exactly 10 units per day."
            self.optimal_profit_baseline = 118000.0 # Adjusted for new $400 margin

        # ==========================================
        # TASK 2: MEDIUM (Holiday Demand Spike)
        # ==========================================
        elif task_id == "task_02_medium":
            self.cash_balance = 50000.0
            self.catalog = {
                "SKU-LAPTOP": {"margin": 400.0, "holding_cost": 2.0, "penalty": 100.0, "order_cost": 800.0}
            }
            self.inventory = {"SKU-LAPTOP": 20}
            self.shipment_pipeline = {"SKU-LAPTOP": {1: 0, 2: 0, 3: 0, 4: 0}}
            self.history = {"sales_yesterday": {"SKU-LAPTOP": 0}, "lost_sales_yesterday": {"SKU-LAPTOP": 0}}
            self.market_signal = "WARNING: Black Friday sale begins on Day 10. Demand will spike from 10 to 40 units per day."
            self.optimal_profit_baseline = 220000.0 # Adjusted for new $400 margin

        # ==========================================
        # TASK 3: HARD (Supply Shock / Delays)
        # ==========================================
        elif task_id == "task_03_hard":
            self.cash_balance = 30000.0
            self.catalog = {
                "SKU-LAPTOP": {"margin": 400.0, "holding_cost": 2.0, "penalty": 100.0, "order_cost": 800.0}
            }
            self.inventory = {"SKU-LAPTOP": 30}
            self.shipment_pipeline = {"SKU-LAPTOP": {1: 0, 2: 0, 3: 0, 4: 0}}
            self.history = {"sales_yesterday": {"SKU-LAPTOP": 0}, "lost_sales_yesterday": {"SKU-LAPTOP": 0}}
            self.market_signal = "WARNING: Global shipping crisis. Standard 2-day shipping is delayed to 4 days. Expedited 1-day shipping costs $100 extra per unit."
            self.optimal_profit_baseline = 105000.0 # Adjusted for new $400 margin
            
        return self.state()

    def state(self) -> SupplyChainObservation:
        warehouse = []
        for pid in self.catalog.keys():
            status = ProductStatus(
                product_id=pid,
                current_stock=self.inventory.get(pid, 0),
                incoming_shipments=self.shipment_pipeline.get(pid, {}),
                sales_yesterday=self.history["sales_yesterday"].get(pid, 0),
                lost_sales_yesterday=self.history["lost_sales_yesterday"].get(pid, 0),
                holding_cost_per_unit=self.catalog[pid]["holding_cost"],
                stockout_penalty_per_unit=self.catalog[pid]["penalty"],
                margin_per_unit=self.catalog[pid]["margin"]
            )
            warehouse.append(status)

        return SupplyChainObservation(
            current_day=self.current_day,
            total_days=self.max_days,
            cash_balance=self.cash_balance,
            warehouse_status=warehouse,
            market_trend_signal=self.market_signal
        )

    def step(self, action: SupplyChainAction) -> Tuple[SupplyChainObservation, float, bool, Dict[str, Any]]:
        reward = 0.0
        
        # 1. Process Arrivals
        for pid in self.catalog.keys():
            arriving_today = self.shipment_pipeline[pid].get(1, 0)
            self.inventory[pid] += arriving_today
            
            new_pipeline = {}
            for day, qty in self.shipment_pipeline[pid].items():
                if day > 1:
                    new_pipeline[day - 1] = qty
            self.shipment_pipeline[pid] = new_pipeline

        # 2. Process New Orders
        for order in action.orders:
            pid = order.product_id
            if pid not in self.catalog or order.quantity <= 0:
                continue

            cost = order.quantity * self.catalog[pid]["order_cost"]
            
            # UPDATED SHIPPING LOGIC: 4 days in crisis, 2 days standard
            if self.current_task_id == "task_03_hard" and not order.expedite_shipping:
                delivery_days = 4 
            else:
                delivery_days = 1 if order.expedite_shipping else 2
                
            # Expedite shipping now costs $100 more per unit (Total $900 per unit)
            if order.expedite_shipping:
                cost += (order.quantity * 100) 
                
            if self.cash_balance >= cost:
                self.cash_balance -= cost
                reward -= cost
                current_queued = self.shipment_pipeline[pid].get(delivery_days, 0)
                self.shipment_pipeline[pid][delivery_days] = current_queued + order.quantity

        # 3. Simulate Daily Demand based on Task
        demand_qty = 10
        if self.current_task_id == "task_02_medium" and 10 <= self.current_day <= 17:
            demand_qty = 40 # Black Friday Spike
            
        actual_demand = {"SKU-LAPTOP": demand_qty} 

        # 4. Fulfill Demand & Calculate Cash Flow (THE ACCOUNTING FIX)
        for pid, demand in actual_demand.items():
            stock = self.inventory[pid]
            sold = min(stock, demand)
            missed = demand - sold
            
            self.inventory[pid] -= sold
            self.history["sales_yesterday"][pid] = sold
            self.history["lost_sales_yesterday"][pid] = missed
            
            margin = self.catalog[pid]["margin"]
            order_cost = self.catalog[pid]["order_cost"]
            
            # Revenue is what the customer actually pays you (Cost + Margin = 1200)
            revenue = sold * (order_cost + margin) 
            
            # Profit for the scoreboard is just the margin (400)
            profit = sold * margin 
            
            holding_cost = self.inventory[pid] * self.catalog[pid]["holding_cost"]
            penalty = missed * self.catalog[pid]["penalty"]
            
            # SCOREKEEPER FIX: The RL Reward needs the full revenue to offset the order cost!
            daily_reward = revenue - holding_cost - penalty
            
            self.cash_balance += revenue  
            reward += daily_reward        

        self.total_profit += reward

        # 5. Advance Time
        self.current_day += 1
        done = self.current_day > self.max_days
        
        info = {"current_profit": self.total_profit}
        return self.state(), reward, done, info

    # ==========================================
    # OPENENV REQUIRED GRADER (0.0 to 1.0)
    # ==========================================
    def get_grader_score(self) -> float:
        """Returns a normalized score between 0.0 and 1.0 based on profit performance."""
        if self.total_profit <= 0:
            return 0.0 # Bankrupt or lost money
            
        score = self.total_profit / self.optimal_profit_baseline
        # Clamp between 0.0 and 1.0
        return max(0.0, min(1.0, score))