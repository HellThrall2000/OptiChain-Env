# OptiChain-Env: AI-Native Supply Chain Optimization

**A Meta OpenEnv Hackathon Submission (2026)**

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![OpenEnv](https://img.shields.io/badge/Meta-OpenEnv-blue?style=for-the-badge)](https://github.com/meta-pytorch/OpenEnv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](https://opensource.org/licenses/MIT)

OptiChain-Env is a high-fidelity inventory management environment built strictly to the **Meta OpenEnv** specification. It challenges AI agents to act as Supply Chain Managers, navigating varying demand signals, shipping disruptions, and strict liquidity constraints over a 30-day simulation.

Unlike standard CLI-only environments, OptiChain ships with a **live dashboard** that visualises the agent's reasoning, JSON action schema, and per-step newsvendor rewards in real-time.

---

## Environment Description

OptiChain simulates a single-SKU retailer running a 30-day episode. Each day the agent observes current stock, cash, shipment pipeline, and yesterday's sales, then decides **how many units to order** and **whether to expedite shipping**. Orders are deducted from cash immediately; inventory arrives after the lead time. At end-of-day, stochastic customer demand is served from on-hand stock — any unmet demand is lost (stockout) and any leftover stock incurs holding cost.

**Core mechanics:**
- **Episode length:** 30 days (fixed)
- **Product:** `SKU-LAPTOP` — order cost $800, margin $400/unit, holding cost $2/unit/day, stockout penalty $100/unit
- **Lead times:** Standard 2 days (4 days under crisis), Expedited 1 day ($100 surcharge/unit)
- **Demand:** Stochastic Poisson-like variation around task-specific base demand
- **Cash constraint:** Orders exceeding available cash are **partially or fully rejected** — the observation reports accepted/rejected quantities so the agent can react

### Reward — Newsvendor Inspired

Each step returns a **bounded `[0, 1]`** reward based on the classic newsvendor overage/underage tradeoff:

| Scenario | Formula | Rewards |
| :--- | :--- | :--- |
| **Overage** (stockout == 0) | `1 - (remaining / stock_before)` | Efficient sell-through |
| **Underage** (stockout > 0) | `1 - (missed / demand)` | High service level |

- `step_reward`: `[0, 1]` — newsvendor score for the day
- `total_reward`: cumulative sum of all step rewards (max 30)
- `grader_score`: `total_reward / max_days` — normalised to `[0, 1]`, this is the **official hackathon score**

The success threshold is **`grader_score >= 0.1`**.

---

## Action Space

```python
class PurchaseOrder(BaseModel):
    product_id: str                 # e.g. "SKU-LAPTOP"
    quantity: int                   # >= 0
    expedite_shipping: bool         # True = 1-day, False = standard

class SupplyChainAction(Action):
    orders: List[PurchaseOrder]     # empty list = do nothing
```

**Example action** (order 15 units, standard shipping):
```json
{ "orders": [ { "product_id": "SKU-LAPTOP", "quantity": 15, "expedite_shipping": false } ] }
```

---

## Observation Space

```python
class ProductStatus(BaseModel):
    product_id: str
    current_stock: int                      # on-hand inventory
    incoming_shipments: Dict[int, int]      # {days_until_arrival: qty}
    sales_yesterday: int
    lost_sales_yesterday: int
    holding_cost_per_unit: float
    stockout_penalty_per_unit: float
    margin_per_unit: float

class SupplyChainObservation(Observation):
    current_day: int                        # 1..30
    total_days: int                         # always 30
    cash_balance: float
    warehouse_status: List[ProductStatus]
    market_trend_signal: str                # task-specific forecast hint
    last_order_accepted: int                # units accepted last step
    last_order_rejected: int                # units rejected (insufficient cash)
    reward: float                           # last step_reward
    done: bool
```

---

## Task Curriculum

| Task ID | Name | Starting Cash | Initial Stock | Challenge |
| :--- | :--- | :--- | :--- | :--- |
| `task_01_easy` | **Stable Demand** | $30,000 | 50 | Constant ~10 units/day demand. Baseline reasoning test. |
| `task_02_medium` | **Holiday Spike** | $50,000 | 20 | Black Friday spike (10 → 40 units/day) on Day 10. Requires pre-loading. |
| `task_03_hard` | **Supply Crisis** | $30,000 | 30 | Standard lead time jumps from 2 to 4 days. Expedited option costs $100/unit extra. |

---

## OpenEnv Endpoints

The environment exposes the full OpenEnv HTTP API:

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `POST` | `/reset` | Start a new episode. Body: `{ "task_id": "...", "seed": 42 }` |
| `POST` | `/step` | Submit a `SupplyChainAction`. Returns observation, reward breakdown, done flag, info |
| `GET`  | `/state` | Current episode metadata without side effects |
| `GET`  | `/grader` | Normalised episode score `[0, 1]` |
| `GET`  | `/health` | Liveness probe for Hugging Face Spaces |
| `GET`  | `/tasks` | List all available tasks and action schema |
| `GET`  | `/schema` | JSON schemas for Action / Observation / Reward / State |
| `GET`  | `/` | Live dashboard UI |
| `POST` | `/demo/step_sim` | Dashboard bridge: run LLM agent → step env → return both |

---

## Project Structure

```text
OptiChain-Env/
├── env/
│   ├── core.py              # Simulation engine (inherits openenv Environment)
│   └── schemas.py           # Pydantic Action/Observation/Reward/State models
├── server/
│   └── app.py               # FastAPI server — canonical OpenEnv entry point
├── static/
│   └── index.html           # Real-time dashboard UI
├── app/
│   └── main.py              # Legacy re-export shim (server.app:app)
├── inference.py             # Multi-agent LLM baseline (Analyst + Executor + guardrails)
├── openenv.yaml             # OpenEnv spec (spec_version: 1)
├── pyproject.toml           # Build config + dependencies
├── requirements.txt         # pip install target
├── Dockerfile               # HF Spaces deployment image
└── .env.example             # LLM API config template
```

---

## Setup Instructions

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/OptiChain-Env.git
cd OptiChain-Env
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python **3.11+** is required.

### 2. Validate the OpenEnv spec

```bash
openenv validate .
```

Expected output: `[OK] : Ready for multi-mode deployment`.

### 3. Run the server locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

- **Dashboard:** http://localhost:7860
- **Swagger UI:** http://localhost:7860/docs

### 4. (Optional) Configure the LLM baseline

The bundled multi-agent baseline in `inference.py` is OpenAI-compatible. Copy `.env.example` to `.env` and set:

```env
API_BASE_URL=https://api.groq.com/openai/v1     # or http://localhost:11434/v1 for Ollama
API_KEY=your_key_here
MODEL_NAME=llama-3.3-70b-versatile               # or llama3.2:3b for Ollama
```

Run the baseline end-to-end:

```bash
python inference.py
```

Structured logs (`[START]`, `[STEP]`, `[END]`) are emitted to stdout.

### 5. Deploy to Hugging Face Spaces

```bash
openenv push --repo-id YOUR_USERNAME/optichain-env
```

Then add `API_BASE_URL`, `API_KEY`, and `MODEL_NAME` as **Space Secrets** (Settings → Repository secrets).

---

## Multi-Agent Baseline

`inference.py` implements a **Planner → Executor → Guardrails** pipeline to keep small/local LLMs reliable:

1. **Analyst agent** — free-form reasoning over observation + order feedback loop
2. **Executor agent** — converts Analyst reasoning to strict `SupplyChainAction` JSON
3. **Python guardrails** — clip orders to `min(affordable_qty, demand_based_useful_qty)` and gate by lead time near episode end

This pattern defeats three failure modes common in small-LLM supply-chain agents:
- **Ghost orders** — silently-rejected orders the LLM believes succeeded
- **Logic-tag dissonance** — reasoning says "order 10" but JSON outputs `quantity: 51`
- **Overstocking** — hoarding cash and inventory far beyond demand horizon

---

## License

MIT — see [LICENSE](LICENSE).
