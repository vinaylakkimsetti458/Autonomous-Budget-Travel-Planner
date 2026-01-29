import streamlit as st
from typing import Dict, Any
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from models.planner_state import PlannerState
from utils.helpers import (
    MAX_REPLANS,
    invoke_llm_with_timeout,
    GROQ_LLM,
)

def budget_review_agent(state: PlannerState) -> PlannerState:
    try:
        remaining = float(state.get('remaining_budget', 0.0))
    except Exception:
        remaining = 0.0

    if remaining >= 0:
        return {
            "is_budget_met": True,
            "messages": [
                SystemMessage(content="Budget review skipped: plan already within budget.")
            ],
            "next_action": "ITINERARY_PLANNER",
        }

    over_budget_by = float(abs(min(0.0, remaining)))

    # ---------- SAFE CACHE KEY ----------
    cache_key = f"budget_review__{int(state.get('budget',0))}_{int(remaining)}"
    cached = st.session_state.get(cache_key)
    if cached:
        return cached

    with st.spinner(f"⚠️ Over Budget by ₹{over_budget_by:,.2f}. Running Budget Review LLM (timed)..."):
        if GROQ_LLM:
            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are a travel budget analyst. The user is over budget by ₹{over_budget}. "
                    "Suggest concise concrete trade-offs focusing on Flights, Hotel, Activities (bulleted list)."
                ),
                (
                    "user",
                    "Current costs: Flight ₹{flight_cost}, Hotel ₹{hotel_cost}, "
                    "Food ₹{food_cost}, Activities ₹{activities_cost}. Original Budget: ₹{budget}."
                )
            ])

            chain = prompt | GROQ_LLM

            ok, raw_or_err = invoke_llm_with_timeout(
                chain,
                {
                    "budget": state.get('budget', 0.0),
                    "flight_cost": state.get('flight_cost', 0.0),
                    "hotel_cost": state.get('accommodation_cost', 0.0),
                    "food_cost": state.get('food_cost', 0.0),
                    "activities_cost": state.get('activities_cost', 0.0),
                    "over_budget": round(over_budget_by, 2),
                },
                timeout=10.0
            )

            if ok:
                suggestion = raw_or_err
            else:
                print("Budget LLM fallback reason:", raw_or_err)
                suggestion = "LLM failed/timed out; consider reducing the largest expense by ~20%."
        else:
            suggestion = "(LLM Disabled) Consider reducing the largest expense by ~20%."

        costs = {
            "flight_cost": float(state.get('flight_cost', 0.0) or 0.0),
            "accommodation_cost": float(state.get('accommodation_cost', 0.0) or 0.0),
            "activities_cost": float(state.get('activities_cost', 0.0) or 0.0),
        }

        largest_cost_key = max(costs, key=costs.get)
        cut_amount = round(costs[largest_cost_key] * 0.20, 2)
        new_cost_value = max(0.0, costs[largest_cost_key] - cut_amount)

        updated_costs = {largest_cost_key: new_cost_value}

        budget_val = float(state.get('budget', 0.0) or 0.0)

        recomputed_sum = (
            updated_costs.get('flight_cost', costs['flight_cost'])
            + updated_costs.get('accommodation_cost', costs['accommodation_cost'])
            + updated_costs.get('activities_cost', costs['activities_cost'])
            + float(state.get('food_cost', 0.0) or 0.0)
        )

        new_remaining_budget = round(budget_val - recomputed_sum, 2)
        is_budget_met_flag = new_remaining_budget >= 0

        messages = [
            SystemMessage(
                content=f"Budget review applied: {largest_cost_key.replace('_',' ')} reduced by ₹{cut_amount:,.2f}."
            ),
            SystemMessage(
                content=f"Recomputed remaining budget: ₹{new_remaining_budget:,.2f}."
            ),
        ]

        out = dict(updated_costs)
        out.update({
    "suggestion": suggestion,
    "action": f"Reduced {largest_cost_key.replace('_',' ')} by ₹{cut_amount:,.2f}.",
    "remaining_budget": new_remaining_budget,
    "is_budget_met": bool(is_budget_met_flag),
    "messages": messages,
    "next_action": "ITINERARY_PLANNER" if is_budget_met_flag else "BUDGET_REVIEW",
})


        st.session_state[cache_key] = out
        return out

