import streamlit as st
from typing import Optional, Dict, Any
from datetime import datetime, date
from langchain_core.messages import SystemMessage
from models.planner_state import PlannerState
from apis.amadeus_api import real_flight_api

def flight_agent(state: PlannerState, task_inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ti = task_inputs or {}
    city_iata = ti.get("city_iata") or state.get("city_iata")
    start_date = ti.get("start_date") or state.get("start_date")
    end_date = ti.get("end_date") or state.get("end_date")
    remaining_budget = ti.get("remaining_budget")
    if remaining_budget is None:
        remaining_budget = state.get("remaining_budget", state.get("budget", 0.0))
    if isinstance(start_date, (datetime, date)):
        start_date = start_date.strftime("%Y-%m-%d")
    if isinstance(end_date, (datetime, date)):
        end_date = end_date.strftime("%Y-%m-%d")
    display_city = state.get("city") or city_iata
    with st.spinner(f"✈️ Booking Flight for {display_city}..."):
        try:
            flight_cost, outbound_details, return_details = real_flight_api(city_iata, start_date, end_date)
        except Exception as e:
            print(f"Flight agent exception: {e}")
            flight_cost, outbound_details, return_details = (0.0, "Simulated Flight/Carrier (Exception)", "Simulated Flight/Carrier (Exception)")
        combined_details = (f"Outbound ({start_date}): {outbound_details} | Return ({end_date}): {return_details}")
        new_remaining = float(remaining_budget) - float(flight_cost)
        trace_entry = {"node": "FLIGHT_AGENT", "city_iata": city_iata, "start_date": start_date, "end_date": end_date, "flight_cost": float(flight_cost)}
        return {"flight_cost": round(float(flight_cost), 2), "flight_details": combined_details, "remaining_budget": round(new_remaining, 2), "messages": [SystemMessage(content=f"Flight cost added: ₹{float(flight_cost):,.2f}")], "next_action": "ACCOMMODATION_AGENT", "trace": [trace_entry]}