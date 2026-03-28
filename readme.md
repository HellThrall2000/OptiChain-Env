# 📦 OptiChain-Env: AI-Native Supply Chain Optimization
**A Meta OpenEnv Hackathon Submission (2026)**

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![OpenAI Compatible](https://img.shields.io/badge/Groq%20/%20Ollama-AI%20Agents-blue?style=for-the-badge)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.style=for-the-badge)](https://opensource.org/licenses/MIT)

**OptiChain-Env** is a high-fidelity inventory management environment built strictly to the **Meta OpenEnv** specification. It challenges AI agents to act as Supply Chain Managers, navigating varying demand signals, shipping disruptions, and strict liquidity constraints to maximize profit.

Unlike standard CLI-only environments, OptiChain includes a **live dashboard** that visualizes the AI's reasoning, JSON schema generation, and environment P&L in real-time.

---

## ✨ Key Features
* **100% OpenEnv Compliant:** Implements standard `/reset`, `/step`, `/state`, and `/grader` endpoints.
* **Multi-Agent Baseline:** Uses a **Planner ➔ Executor** delegation pattern to separate strategic reasoning from strict Pydantic JSON formatting.
* **Live UI Dashboard:** A completely decoupled frontend served via FastAPI that bridges LLM logic with the simulation engine for real-time visualization.
* **LLM Agnostic:** Designed to work with Groq (Llama-3.3-70B) for high-speed cloud inference or Ollama (Llama-3.2) for local testing.

---

## 🚀 The Curriculum
The environment tests an agent's reasoning across three distinct difficulty levels. Every laptop costs **$800** to order, with a standard holding cost and stockout penalty.

| Task ID | Name | Starting Cash | The Challenge |
| :--- | :--- | :--- | :--- |
| `task_01_easy` | **Stable Demand** | $30,000 | Baseline testing with a constant 10 units/day demand. |
| `task_02_medium` | **Holiday Spike** | $50,000 | High-volume demand spikes (40 units/day) requiring pre-loading. |
| `task_03_hard` | **Supply Crisis** | $30,000 | Logistical breakdown: Shipping delay increases from 3 to 10 days. |

---

## 🧠 Architecture

```text
OptiChain-Env/
├── app/
│   └── main.py          # FastAPI Server & UI Bridge Endpoint
├── env/
│   ├── core.py          # The Mathematical Supply Chain Engine
│   └── schemas.py       # Pydantic Action/Observation Models
├── static/
│   └── index.html       # The Real-Time Dashboard UI
├── baseline.py          # Multi-Agent Logic (Analyst & Executor)
├── openenv.yaml         # Meta OpenEnv Configuration
└── requirements.txt     # Environment Dependencies