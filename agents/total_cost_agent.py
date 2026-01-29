from langchain_core.messages import SystemMessage
from models.planner_state import PlannerState

def total_cost_check(state: PlannerState) -> dict:
    if state.get("remaining_budget", 0.0) < 0:
        return {
            "messages": [SystemMessage(content="Total cost exceeded budget.")],
            "next_action": "BUDGET_REVIEW"
        }
    else:
        return {
            "messages": [SystemMessage(content="Total cost within budget.")],
            "next_action": "ITINERARY_PLANNER"
        }



