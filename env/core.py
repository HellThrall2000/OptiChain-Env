import random
import uuid

try:
    from openenv.core.env_server import Environment
except ImportError:
    from core.env_server import Environment

from env.schemas import (
    SupplyChainAction,
    SupplyChainObservation,
    SupplyChainState,
    ProductStatus,
)


class SupplyChainEnv(Environment):
    """
    OpenEnv-compliant supply chain inventory management environment.

    Inherits from openenv.core.env_server.Environment.
    The agent acts as a Supply Chain Manager, placing daily purchase orders
    to maximize profit across a 30-day simulation.
    """

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self):
        super().__init__()
        self.current_task_id: str = ""
        self.max_days: int = 30
        self.current_day: int = 0
        self.cash_balance: float = 0.0
        self.inventory: dict = {}
        self.shipment_pipeline: dict = {}
        self.history: dict = {"sales_yesterday": {}, "lost_sales_yesterday": {}}
        self.catalog: dict = {}
        self.market_signal: str = ""
        self.total_profit: float = 0.0
        self.optimal_profit_baseline: float = 0.0
        # OpenEnv State container (tracks episode metadata)
        self._state = SupplyChainState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
        )

    def reset(self, task_id: str = "task_01_easy") -> SupplyChainObservation:
        """Reset the environment and load a specific task. Returns initial observation."""
        self.current_task_id = task_id
        self.current_day = 1
        self.total_profit = 0.0

        # Reset the OpenEnv State container
        self._state = SupplyChainState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            current_task_id=task_id,
        )

        # ==========================================
        # TASK 1: EASY (Stable Demand)
        # ==========================================
        if task_id == "task_01_easy":
            self.cash_balance = 30000.0
            self.catalog = {
                "SKU-LAPTOP": {"margin": 400.0, "holding_cost": 2.0, "penalty": 100.0, "order_cost": 800.0}
            }
            self.inventory = {"SKU-LAPTOP": 50}
            self.shipment_pipeline = {"SKU-LAPTOP": {1: 0, 2: 0, 3: 0, 4: 0}}
            self.history = {"sales_yesterday": {"SKU-LAPTOP": 0}, "lost_sales_yesterday": {"SKU-LAPTOP": 0}}
            self.market_signal = "Demand is stable at exactly 10 units per day."
            self.optimal_profit_baseline = 118000.0

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
            self.optimal_profit_baseline = 220000.0

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
            self.optimal_profit_baseline = 105000.0

        return self._build_observation(reward=0.0, done=False)

    def step(self, action: SupplyChainAction) -> SupplyChainObservation:
        """Execute one day of the simulation. Returns observation with reward and done."""
        step_reward = 0.0

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

            # 4-day standard shipping during crisis, 2-day otherwise
            if self.current_task_id == "task_03_hard" and not order.expedite_shipping:
                delivery_days = 4
            else:
                delivery_days = 1 if order.expedite_shipping else 2

            # Expedite shipping costs $100 extra per unit
            if order.expedite_shipping:
                cost += order.quantity * 100

            if self.cash_balance >= cost:
                self.cash_balance -= cost
                step_reward -= cost
                current_queued = self.shipment_pipeline[pid].get(delivery_days, 0)
                self.shipment_pipeline[pid][delivery_days] = current_queued + order.quantity

        # 3. Simulate Daily Demand (stochastic)
        if self.current_task_id == "task_01_easy":
            demand_qty = random.randint(8, 12)
        elif self.current_task_id == "task_02_medium":
            demand_qty = random.randint(35, 45) if 10 <= self.current_day <= 17 else random.randint(8, 12)
        elif self.current_task_id == "task_03_hard":
            demand_qty = random.randint(5, 20)
        else:
            demand_qty = 10

        actual_demand = {"SKU-LAPTOP": demand_qty}

        # 4. Fulfill Demand & Calculate Cash Flow
        for pid, demand in actual_demand.items():
            stock = self.inventory[pid]
            sold = min(stock, demand)
            missed = demand - sold

            self.inventory[pid] -= sold
            self.history["sales_yesterday"][pid] = sold
            self.history["lost_sales_yesterday"][pid] = missed

            margin = self.catalog[pid]["margin"]
            order_cost = self.catalog[pid]["order_cost"]

            # Revenue = selling price × units sold (cost + margin = 1200)
            revenue = sold * (order_cost + margin)
            holding_cost = self.inventory[pid] * self.catalog[pid]["holding_cost"]
            penalty = missed * self.catalog[pid]["penalty"]

            daily_reward = revenue - holding_cost - penalty
            self.cash_balance += revenue
            step_reward += daily_reward

        self.total_profit += step_reward

        # 5. Advance Time
        self.current_day += 1
        done = self.current_day > self.max_days

        # Sync OpenEnv state container
        self._state.step_count += 1
        self._state.current_task_id = self.current_task_id
        self._state.total_profit = self.total_profit
        self._state.optimal_profit_baseline = self.optimal_profit_baseline
        self._state.grader_score = self.get_grader_score()

        return self._build_observation(reward=step_reward, done=done)

    @property
    def state(self) -> SupplyChainState:
        """OpenEnv-required property: returns current episode metadata."""
        self._state.current_task_id = self.current_task_id
        self._state.total_profit = self.total_profit
        self._state.optimal_profit_baseline = self.optimal_profit_baseline
        self._state.grader_score = self.get_grader_score()
        return self._state

    def _build_observation(self, reward: float = 0.0, done: bool = False) -> SupplyChainObservation:
        """Build the current observation snapshot."""
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
                margin_per_unit=self.catalog[pid]["margin"],
            )
            warehouse.append(status)

        return SupplyChainObservation(
            current_day=self.current_day,
            total_days=self.max_days,
            cash_balance=self.cash_balance,
            warehouse_status=warehouse,
            market_trend_signal=self.market_signal,
            reward=reward,
            done=done,
        )

    # ==========================================
    # OPENENV REQUIRED GRADER (0.0 to 1.0)
    # ==========================================
    def get_grader_score(self) -> float:
        """Returns a normalized score between 0.0 and 1.0 based on profit performance."""
        if self.total_profit <= 0:
            return 0.0
        score = self.total_profit / self.optimal_profit_baseline
        return max(0.0, min(1.0, score))
