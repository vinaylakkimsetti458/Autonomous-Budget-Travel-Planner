import streamlit as st
from typing import Optional, Dict, Any
from datetime import datetime, date
from langchain_core.messages import SystemMessage
from models.planner_state import PlannerState
from apis.amadeus_api import real_hotel_api

def accommodation_agent(state: PlannerState, task_inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    
    # Use task-specific inputs if provided, otherwise fall back to global state
    ti = task_inputs or {}
    city_iata = ti.get("city_iata") or state.get("city_iata")
    start_date = ti.get("start_date") or state.get("start_date")
    end_date = ti.get("end_date") or state.get("end_date")

    # Duration is required for pricing; default to 1 day if missing
    duration_days = int(ti.get("duration_days") or state.get("duration_days", 1))

    # Remaining budget may come from task inputs or previous agent output
    remaining_budget = ti.get("remaining_budget")

    # Normalize date formats for API compatibility
    if remaining_budget is None:
        remaining_budget = state.get("remaining_budget", state.get("budget", 0.0))
    if isinstance(start_date, (datetime, date)):
        start_date = start_date.strftime("%Y-%m-%d")
    if isinstance(end_date, (datetime, date)):
        end_date = end_date.strftime("%Y-%m-%d")

    # Show progress spinner in Streamlit UI
    with st.spinner(f"🏨 Finding Accommodation in {state.get('city', city_iata)}..."):
        # Call real hotel API to get cost and details
        try:
            accommodation_cost, accommodation_details = real_hotel_api(city_iata, start_date, end_date, duration_days)
        except Exception as e:
            # Graceful fallback in case of API failure
            print(f"Accommodation agent exception: {e}")
            accommodation_cost = 0.0
            accommodation_details = "Simulated Accommodation (Exception)"

        # Update remaining budget after accommodation expense
        new_remaining = float(remaining_budget) - float(accommodation_cost)

        # Trace entry for debugging / observability
        trace_entry = {"node": "ACCOMMODATION_AGENT", "city_iata": city_iata, "nights": max(1, duration_days - 1), "accommodation_cost": float(accommodation_cost)}

        # Return structured output for downstream agents
        return {"accommodation_cost": round(float(accommodation_cost), 2), "accommodation_details": accommodation_details, "remaining_budget": round(new_remaining, 2), "messages": [SystemMessage(content=f"Accommodation cost added: ₹{float(accommodation_cost):,.2f}")], "next_action": "FOOD_AGENT", "trace": [trace_entry]}
