import streamlit as st
from typing import Optional, Dict, Any
from datetime import datetime, date
from langchain_core.messages import SystemMessage
from models.planner_state import PlannerState
from apis.amadeus_api import real_hotel_api

def accommodation_agent(state: PlannerState, task_inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ti = task_inputs or {}
    city_iata = ti.get("city_iata") or state.get("city_iata")
    start_date = ti.get("start_date") or state.get("start_date")
    end_date = ti.get("end_date") or state.get("end_date")
    duration_days = int(ti.get("duration_days") or state.get("duration_days", 1))
    remaining_budget = ti.get("remaining_budget")
    if remaining_budget is None:
        remaining_budget = state.get("remaining_budget", state.get("budget", 0.0))
    if isinstance(start_date, (datetime, date)):
        start_date = start_date.strftime("%Y-%m-%d")
    if isinstance(end_date, (datetime, date)):
        end_date = end_date.strftime("%Y-%m-%d")
    with st.spinner(f"🏨 Finding Accommodation in {state.get('city', city_iata)}..."):
        try:
            accommodation_cost, accommodation_details = real_hotel_api(city_iata, start_date, end_date, duration_days)
        except Exception as e:
            print(f"Accommodation agent exception: {e}")
            accommodation_cost = 0.0
            accommodation_details = "Simulated Accommodation (Exception)"
        new_remaining = float(remaining_budget) - float(accommodation_cost)
        trace_entry = {"node": "ACCOMMODATION_AGENT", "city_iata": city_iata, "nights": max(1, duration_days - 1), "accommodation_cost": float(accommodation_cost)}
        return {"accommodation_cost": round(float(accommodation_cost), 2), "accommodation_details": accommodation_details, "remaining_budget": round(new_remaining, 2), "messages": [SystemMessage(content=f"Accommodation cost added: ₹{float(accommodation_cost):,.2f}")], "next_action": "FOOD_AGENT", "trace": [trace_entry]}
