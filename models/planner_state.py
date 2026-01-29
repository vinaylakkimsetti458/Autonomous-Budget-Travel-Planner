from typing import TypedDict, List
from langchain_core.messages import SystemMessage, HumanMessage

class PlannerState(TypedDict):
    city: str
    city_iata: str
    start_date: str
    end_date: str
    budget: float
    duration_days: int
    messages: List[SystemMessage | HumanMessage]
    remaining_budget: float
    flight_cost: float
    accommodation_cost: float
    food_cost: float
    activities_cost: float
    flight_details: str
    accommodation_details: str
    food_itinerary: List[str]
    activities_plan: List[str]
    itinerary_draft: str
    itinerary_extras: str
    is_budget_met: bool
    next_action: str
