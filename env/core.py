import random
import uuid
from openenv.core.env_server import Environment

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
        self.total_reward: float = 0.0  # cumulative sum of per-step newsvendor rewards
        # OpenEnv State container (tracks episode metadata)
        self._state = SupplyChainState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
        )

    def reset(self, task_id: str = "task_01_easy") -> SupplyChainObservation:
        """Reset the environment and load a specific task. Returns initial observation."""
        self.current_task_id = task_id
        self.current_day = 1
        self.total_reward = 0.0

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

        return self._build_observation(reward=0.0, done=False)

    def step(self, action: SupplyChainAction) -> SupplyChainObservation:
        """Execute one day of the simulation. Returns observation with reward and done."""
        # 1. Process Arrivals
        for pid in self.catalog.keys():
            arriving_today = self.shipment_pipeline[pid].get(1, 0)
            self.inventory[pid] += arriving_today
            new_pipeline = {}
            for day, qty in self.shipment_pipeline[pid].items():
                if day > 1:
                    new_pipeline[day - 1] = qty
            self.shipment_pipeline[pid] = new_pipeline

        # 2. Process New Orders (track accepted vs rejected)
        self.last_accepted_qty = 0
        self.last_rejected_qty = 0
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
                current_queued = self.shipment_pipeline[pid].get(delivery_days, 0)
                self.shipment_pipeline[pid][delivery_days] = current_queued + order.quantity
                self.last_accepted_qty += order.quantity
            else:
                self.last_rejected_qty += order.quantity

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

        # 4. Fulfill Demand, update cash, compute per-SKU newsvendor reward
        product_rewards = []
        for pid, demand in actual_demand.items():
            stock_before = self.inventory[pid]       # total stock before selling
            sold = min(stock_before, demand)
            missed = demand - sold
            remaining = stock_before - sold           # stock left after selling

            # Update inventory and history
            self.inventory[pid] = remaining
            self.history["sales_yesterday"][pid] = sold
            self.history["lost_sales_yesterday"][pid] = missed

            # Cash flow: revenue from sales (still drives budget for future orders)
            revenue = sold * (self.catalog[pid]["order_cost"] + self.catalog[pid]["margin"])
            self.cash_balance += revenue

            # ------------------------------------------------------------------
            # Newsvendor step reward — bounded [0, 1]
            #
            # Scenario 1 — Overage (all demand met, stock remains):
            #   reward = 1 - (remaining / stock_before)
            #   → rewards efficient use of inventory; perfect sell-through = 1.0
            #
            # Scenario 2 — Underage (stockout, unmet demand):
            #   reward = 1 - (missed / demand)
            #   → rewards high service level; zero missed orders = 1.0
            # ------------------------------------------------------------------
            if missed == 0:
                # Overage or perfect match
                product_reward = 1.0 - (remaining / stock_before) if stock_before > 0 else 1.0
            else:
                # Underage (demand > 0 guaranteed since missed > 0)
                product_reward = 1.0 - (missed / demand)

            product_rewards.append(product_reward)

        # Average across SKUs (scales cleanly to multi-product tasks)
        step_reward = sum(product_rewards) / len(product_rewards) if product_rewards else 0.0
        self.total_reward += step_reward

        # 5. Advance Time
        self.current_day += 1
        done = self.current_day > self.max_days

        # Sync OpenEnv state container
        self._state.step_count += 1
        self._state.current_task_id = self.current_task_id
        self._state.total_reward = self.total_reward
        self._state.grader_score = self.get_grader_score()

        return self._build_observation(reward=step_reward, done=done)

    @property
    def state(self) -> SupplyChainState:
        """OpenEnv-required property: returns current episode metadata."""
        self._state.current_task_id = self.current_task_id
        self._state.total_reward = self.total_reward
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
            last_order_accepted=getattr(self, "last_accepted_qty", 0),
            last_order_rejected=getattr(self, "last_rejected_qty", 0),
            reward=reward,
            done=done,
        )

    # ==========================================
    # OPENENV REQUIRED GRADER (0.0 to 1.0)
    # ==========================================
    def get_grader_score(self) -> float:
        """
        Returns a normalized score in [0.0, 1.0].
        total_reward = sum of per-step newsvendor rewards / max_days
        Each step reward is already in [0, 1], so dividing by max_days keeps
        the episode score in [0, 1] with partial-progress credit throughout.
        """
        return max(0.0, min(1.0, self.total_reward / self.max_days))
