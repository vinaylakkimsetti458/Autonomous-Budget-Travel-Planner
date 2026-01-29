import streamlit as st
from typing import Optional, Dict, Any
from datetime import datetime, date
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate 
from models.planner_state import PlannerState
from utils.helpers import calculate_duration, GROQ_LLM, safe_invoke_planner

def planner_agent(user_goal: Dict[str, Any], manifest: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    # deterministic fallback plan (same as your original fallback)
    FALLBACK_PLAN = {
        "plan_id": f"fallback_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        "tasks": [
            {"task_id": "t1", "node": "FLIGHT_AGENT", "inputs": {"city_iata": user_goal.get("city_iata"), "start_date": user_goal.get("start_date"), "end_date": user_goal.get("end_date"), "budget": user_goal.get("budget")}, "parallel": False, "on_success": "t2", "on_failure": "BUDGET_REVIEW"},
            {"task_id": "t2", "node": "ACCOMMODATION_AGENT", "inputs": {"city_iata": user_goal.get("city_iata"), "start_date": user_goal.get("start_date"), "end_date": user_goal.get("end_date"), "duration_days": user_goal.get("duration_days"), "budget": user_goal.get("budget")}, "parallel": True, "on_success": "t3", "on_failure": "BUDGET_REVIEW"},
            {"task_id": "t3", "node": "FOOD_AGENT", "inputs": {"city": user_goal.get("city"), "duration_days": user_goal.get("duration_days"), "budget": user_goal.get("budget")}, "parallel": True, "on_success": "t4", "on_failure": "t4"},
            {"task_id": "t4", "node": "ACTIVITIES_AGENT", "inputs": {"city": user_goal.get("city"), "duration_days": user_goal.get("duration_days"), "remaining_budget": user_goal.get("budget")}, "parallel": True, "on_success": "t5", "on_failure": "BUDGET_REVIEW"},
            {"task_id": "t5", "node": "TOTAL_COST_CHECK", "inputs": {}, "parallel": False, "on_success": "ITINERARY_PLANNER", "on_failure": "BUDGET_REVIEW"},
        ]
    }

    # If LLM not configured, return fallback quickly
    if not GROQ_LLM:
        return FALLBACK_PLAN

    # Stronger prompt: include schema + single short example + explicit fallback token
    planner_prompt = """
You are an orchestration planner. Input: user_goal, agent manifest, context.
Return ONLY valid JSON following this schema EXACTLY:
{ "plan_id": string, "tasks": [ { "task_id": string, "node": string, "inputs": object, "parallel": boolean, "on_success": string, "on_failure": string } ] }
MAX tasks: 8.
If you cannot produce valid JSON, return {"error":"cannot_generate"}.
SAMPLE:
{"plan_id":"ex1","tasks":[{"task_id":"t1","node":"FLIGHT_AGENT","inputs":{},"parallel":false,"on_success":"t2","on_failure":"BUDGET_REVIEW"}]}
"""
    try:
        chain_prompt = ChatPromptTemplate.from_messages([("system", planner_prompt), ("user", "USER_GOAL: {goal}\nMANIFEST: {manifest}\nCONTEXT: {context}\nReturn JSON only.")])
        result = safe_invoke_planner(chain_prompt, {"goal": user_goal, "manifest": manifest, "context": context})
        if not result["ok"]:
            # Log structured error and return fallback (no recursion)
            print("[Planner] safe_invoke_planner failed:", result.get("error"), result.get("last_raw"))
            return FALLBACK_PLAN

        plan = result["plan"]

        # Quick structural checks
        if not isinstance(plan, dict) or "tasks" not in plan or not isinstance(plan.get("tasks"), list):
            print("[Planner] Model returned invalid schema, using fallback. Plan preview:", str(plan)[:1000])
            return FALLBACK_PLAN

        if len(plan.get("tasks", [])) > 8:
            print("[Planner] Model produced >8 tasks; using fallback.")
            return FALLBACK_PLAN

        return plan

    except Exception as e:
        # Catch-all: log and use fallback (no recursion)
        print("[Planner] Unexpected error in call_planner_llm:", repr(e))
        return FALLBACK_PLAN