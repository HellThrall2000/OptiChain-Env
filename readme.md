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
```

---

## PR Request Guideline
Step 1: Create the file
In your root directory, create a new folder named .github. Inside that folder, create a file named PULL_REQUEST_TEMPLATE.md.

Your folder structure should look like this:

Plaintext
OptiChain-Env/
├── .github/
│   └── PULL_REQUEST_TEMPLATE.md
├── app/
...
Step 2: Paste the Template
Paste this markdown code into the PULL_REQUEST_TEMPLATE.md file:

Markdown
## 📝 Description
* ## 🔄 Type of Change
- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] ✨ New feature (non-breaking change which adds functionality)
- [ ] ♻️ Refactoring (code changes that neither fix a bug nor add a feature)
- [ ] 🎨 UI/Frontend update (changes Svelte/HTML/CSS without touching core logic)

## ✅ OpenEnv Compliance Checklist
- [ ] My code strictly adheres to the Meta OpenEnv specification.
- [ ] I have not broken the `/reset`, `/step`, or `/state` API endpoints.
- [ ] I have verified that `env.get_grader_score()` still returns a float between 0.0 and 1.0.
- [ ] I ran `python baseline.py` locally and the LLM agent completed the simulation without crashing.

## 📸 Screenshots / Outputs (if applicable)
Step 3: Commit and Push
Commit this new file to your main branch before you lock it down:

Bash
git add .github/PULL_REQUEST_TEMPLATE.md
git commit -m "chore: add PR template for strict hackathon workflow"
git push origin main
