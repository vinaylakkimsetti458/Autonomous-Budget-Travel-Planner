import streamlit as st
from typing import Optional, Dict, Any, List
from langchain_core.messages import SystemMessage
from models.planner_state import PlannerState
from apis.amadeus_api import real_activities_budget_and_list
from utils.helpers import (
    build_daywise_activities_plan,
    flatten_activities_plan_for_prompt,
    compute_total_used_activities_cost,
    mock_activities_agent,
)

def activities_agent(state: PlannerState, task_inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ti = task_inputs or {}
    city = ti.get("city") or state.get("city")
    duration_days = int(ti.get("duration_days") or state.get("duration_days", 1))
    remaining_budget = ti.get("remaining_budget")
    if remaining_budget is None:
        remaining_budget = state.get("remaining_budget", state.get("budget", 0.0))
    with st.spinner("🗺️ Estimating Activity & Local Travel Budget..."):
        try:
            est_cost, details, activities_list = real_activities_budget_and_list(city, duration_days, remaining_budget)
        except Exception as e:
            print(f"Activities agent exception: {e}")
            est_cost, details = mock_activities_agent(city, duration_days, remaining_budget)
            activities_list = []
        daywise_plan = build_daywise_activities_plan(activities_list, num_days=duration_days, slots=("morning", "afternoon", "evening"))
        activities_plan_lines = flatten_activities_plan_for_prompt(daywise_plan)
        real_total_cost = compute_total_used_activities_cost(daywise_plan)
        if real_total_cost <= 0:
            activities_cost = float(est_cost)
            cost_note = "(estimated)"
        else:
            activities_cost = float(real_total_cost)
            cost_note = "(actual sum)"
        new_remaining = float(remaining_budget) - float(activities_cost)
        trace_entry = {"node": "ACTIVITIES_AGENT", "city": city, "duration_days": int(duration_days), "activities_count": len(activities_list), "activities_cost": float(activities_cost)}
        details_msg = f"Activities cost added: ₹{float(activities_cost):,.2f} {cost_note} — {details if isinstance(details, str) else ''}"
        return {"activities_cost": round(float(activities_cost), 2), "remaining_budget": round(new_remaining, 2), "activities_plan": activities_plan_lines, "messages": [SystemMessage(content=details_msg)], "next_action": "TOTAL_COST_CHECK", "trace": [trace_entry]}