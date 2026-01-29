import streamlit as st
from typing import Optional, Dict, Any
from langchain_core.messages import SystemMessage
from models.planner_state import PlannerState
from apis.geoapify_api import real_food_api
from utils.helpers import MEAL_COST_TIERS


def food_agent(state: PlannerState, task_inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    # Use task-specific inputs if provided; otherwise fall back to planner state
    ti = task_inputs or {}

    # Resolve destination city name
    city = ti.get("city") or state.get("city")

    # Determine trip duration in days (default to 1 if missing)
    duration_days = int(ti.get("duration_days") or state.get("duration_days", 1))

    # Fetch remaining budget from task inputs or state
    remaining_budget = ti.get("remaining_budget")
    if remaining_budget is None:
        remaining_budget = state.get("remaining_budget", state.get("budget", 0.0))

    # Show progress indicator while planning meals and restaurant suggestions
    with st.spinner("🍔 Planning Meals & Restaurant Suggestions..."):
        try:
            # Call Geoapify API to estimate food cost and generate meal itinerary
            food_cost, food_itinerary = real_food_api(city, duration_days)
        except Exception as e:
            # Graceful fallback if food API fails
            print(f"Food agent exception: {e}")
            food_cost = 0.0
            food_itinerary = [
                f"Day 1 Lunch (Medium - ₹{MEAL_COST_TIERS['Medium']:.0f}): Casual Dining"
            ]

        # Update remaining budget after deducting food cost
        new_remaining = float(remaining_budget) - float(food_cost)

        # Create trace entry for observability and debugging
        trace_entry = {
            "node": "FOOD_AGENT",
            "city": city,
            "duration_days": int(duration_days),
            "food_cost": float(food_cost)
        }

        # Return structured output for the next agent in the pipeline
        return {
            "food_cost": round(float(food_cost), 2),
            "food_itinerary": food_itinerary,
            "remaining_budget": round(new_remaining, 2),
            "messages": [
                SystemMessage(
                    content=f"Food cost added: ₹{float(food_cost):,.2f}"
                )
            ],
            "next_action": "ACTIVITIES_AGENT",
            "trace": [trace_entry]
        }

