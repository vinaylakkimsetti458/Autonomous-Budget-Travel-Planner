import streamlit as st
from typing import Optional, Dict, Any
from langchain_core.messages import SystemMessage
from models.planner_state import PlannerState
from apis.geoapify_api import real_food_api
from utils.helpers import MEAL_COST_TIERS

def food_agent(state: PlannerState, task_inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ti = task_inputs or {}
    city = ti.get("city") or state.get("city")
    duration_days = int(ti.get("duration_days") or state.get("duration_days", 1))
    remaining_budget = ti.get("remaining_budget")
    if remaining_budget is None:
        remaining_budget = state.get("remaining_budget", state.get("budget", 0.0))
    with st.spinner("🍔 Planning Meals & Restaurant Suggestions..."):
        try:
            food_cost, food_itinerary = real_food_api(city, duration_days)
        except Exception as e:
            print(f"Food agent exception: {e}")
            food_cost = 0.0
            food_itinerary = [f"Day 1 Lunch (Medium - ₹{MEAL_COST_TIERS['Medium']:.0f}): Casual Dining"]
        new_remaining = float(remaining_budget) - float(food_cost)
        trace_entry = {"node": "FOOD_AGENT", "city": city, "duration_days": int(duration_days), "food_cost": float(food_cost)}
        return {"food_cost": round(float(food_cost), 2), "food_itinerary": food_itinerary, "remaining_budget": round(new_remaining, 2), "messages": [SystemMessage(content=f"Food cost added: ₹{float(food_cost):,.2f}")], "next_action": "ACTIVITIES_AGENT", "trace": [trace_entry]}
