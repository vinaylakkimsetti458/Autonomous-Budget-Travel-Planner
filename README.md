# 🌍 Autonomous Budget Travel Planner

An **autonomous, budget-aware travel planning system** built using a **multi-agent architecture**.  
The system intelligently plans an end-to-end trip — including flights, accommodation, food, activities, total cost evaluation, budget optimization, and itinerary generation — all within a user-defined budget.

This project demonstrates how **LLM-powered agents can collaborate, replan, and self-correct** to solve a real-world planning problem.

---

## ✨ Key Features

- 🧠 **Autonomous Multi-Agent System**
- 💰 **Budget-Aware Cost Optimization**
- 🔁 **Planner–Replanner Loop for Fault Tolerance**
- 🧮 **Centralized Total Cost Aggregation**
- 🗓️ **Day-by-Day Itinerary Generation**
- 📧 **Email Delivery of Final Plan**
- 🌐 **Interactive Streamlit Web Interface**

---

## 🧠 Agent Architecture Overview

The system is composed of specialized agents, each responsible for a specific task in the travel-planning pipeline.

| Agent | Responsibility |
|------|---------------|
| **Planner Agent** | Creates the dynamic execution plan and decides which agents to invoke |
| **Replanner Agent** | Rebuilds the plan when failures, invalid outputs, or budget violations occur |
| **Flight Agent** | Estimates the cheapest round-trip flight cost |
| **Accommodation Agent** | Calculates hotel/accommodation expenses |
| **Food Agent** | Plans daily meals and estimates food costs |
| **Activities Agent** | Selects activities within the remaining budget |
| **Total Cost Agent** | Aggregates costs from all agents into a final trip cost |
| **Budget Review Agent** | Optimizes or reduces costs when budget limits are exceeded |
| **Itinerary Planner Agent** | Generates a detailed day-by-day itinerary |
| **Email Agent** | Sends the final itinerary and cost summary to the user via email |

---

## 🏗️ System Architecture

```text
User Input (Streamlit UI)
        |
        v
🧠 Planner Agent
(Dynamic task planning via LangGraph)
        |
        v
+--------------------------------------+
|        Parallel Cost Agents           |
|--------------------------------------|
| ✈️ Flight Agent                       |
| 🏨 Accommodation Agent                |
| 🍽️ Food Agent                         |
| 🎡 Activities Agent                   |
+--------------------------------------+
        |
        v
🧮 Total Cost Agent
(Aggregates all costs)
        |
        v
💰 Budget Review Agent
        |
        +--> Budget exceeded?
        |       |
        |       v
        |   🔁 Replanner Agent
        |   (Rebuilds execution plan)
        |
        v
🗓️ Itinerary Planner Agent
        |
        v
📧 Email Agent
        |
        v
Final Travel Plan Delivered
