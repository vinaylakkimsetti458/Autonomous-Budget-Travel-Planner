import streamlit as st
from typing import Optional, Dict, Any
from datetime import datetime, date
from langchain_core.messages import SystemMessage
from models.planner_state import PlannerState
from apis.amadeus_api import real_flight_api


def flight_agent(state: PlannerState, task_inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    # Use task-level inputs if provided; otherwise rely on shared planner state
    ti = task_inputs or {}

    # Resolve destination airport IATA code
    city_iata = ti.get("city_iata") or state.get("city_iata")

    # Resolve trip start and end dates
    start_date = ti.get("start_date") or state.get("start_date")
    end_date = ti.get("end_date") or state.get("end_date")

    # Fetch remaining budget from task inputs or state
    remaining_budget = ti.get("remaining_budget")
    if remaining_budget is None:
        remaining_budget = state.get("remaining_budget", state.get("budget", 0.0))

    # Normalize date objects into API-compatible string format
    if isinstance(start_date, (datetime, date)):
        start_date = start_date.strftime("%Y-%m-%d")
    if isinstance(end_date, (datetime, date)):
        end_date = end_date.strftime("%Y-%m-%d")

    # Prefer user-friendly city name for UI display
    display_city = state.get("city") or city_iata

    # Show progress indicator while fetching flight details
    with st.spinner(f"✈️ Booking Flight for {display_city}..."):
        try:
            # Call flight API to retrieve cost and route details
            flight_cost, outbound_details, return_details = real_flight_api(
                city_iata, start_date, end_date
            )
        except Exception as e:
            # Graceful fallback if flight API fails
            print(f"Flight agent exception: {e}")
            flight_cost, outbound_details, return_details = (
                0.0,
                "Simulated Flight/Carrier (Exception)",
                "Simulated Flight/Carrier (Exception)"
            )

        # Combine outbound and return flight details into a single readable string
        combined_details = (
            f"Outbound ({start_date}): {outbound_details} | "
            f"Return ({end_date}): {return_details}"
        )

        # Update remaining budget after deducting flight cost
        new_remaining = float(remaining_budget) - float(flight_cost)

        # Create trace entry for observability and debugging
        trace_entry = {
            "node": "FLIGHT_AGENT",
            "city_iata": city_iata,
            "start_date": start_date,
            "end_date": end_date,
            "flight_cost": float(flight_cost)
        }

        # Return structured output for downstream agents
        return {
            "flight_cost": round(float(flight_cost), 2),
            "flight_details": combined_details,
            "remaining_budget": round(new_remaining, 2),
            "messages": [
                SystemMessage(
                    content=f"Flight cost added: ₹{float(flight_cost):,.2f}"
                )
            ],
            "next_action": "ACCOMMODATION_AGENT",
            "trace": [trace_entry]
        }
