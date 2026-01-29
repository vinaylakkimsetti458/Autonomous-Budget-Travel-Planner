import json
import streamlit as st
from typing import Dict, Any
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from utils.helpers import GROQ_LLM, safe_invoke_planner

def replanner_agent(
    failure_context: Dict[str, Any],
    manifest: Dict[str, Any],
    current_plan: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Replanner with SAME functionality.

    Order:
      1) Retry failed task (if allowed)
      2) Use fallback provider (if exists)
      3) Ask LLM to rebuild plan
      4) If still invalid → remove failed task
    """

    failing_id = failure_context.get("failing_task_id")

    # ----------------- helpers -----------------

    def wrap(tasks):
        return {
            "plan_id": f"replan_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "tasks": tasks,
        }

    def find_task(tasks, task_id):
        return next((t for t in tasks if t.get("task_id") == task_id), None)

    def make_retry(task):
        return {
            "task_id": task["task_id"] + "_retry",
            "node": task["node"],
            "inputs": {**task.get("inputs", {}), "_retry_relax": True},
            "parallel": False,
            "on_success": task["on_success"],
            "on_failure": "BUDGET_REVIEW",
        }

    def make_fallback(task, fallback_node):
        return {
            "task_id": task["task_id"] + "_fallback",
            "node": fallback_node,
            "inputs": {**task.get("inputs", {}), "_from_fallback": True},
            "parallel": False,
            "on_success": task["on_success"],
            "on_failure": "BUDGET_REVIEW",
        }

    # ----------------- deterministic mode -----------------

    if not GROQ_LLM:
        failed_task = find_task(current_plan.get("tasks", []), failing_id)
        if not failed_task:
            return wrap(current_plan.get("tasks", []))

        node = failed_task["node"]

        # 1️⃣ retry
        max_retries = manifest.get(node, {}).get("retry_strategy", {}).get("max_retries", 0)
        if max_retries > 0:
            return wrap([
                make_retry(t) if t["task_id"] == failing_id else t
                for t in current_plan["tasks"]
            ])

        # 2️⃣ fallback provider
        fallback_nodes = manifest.get(node, {}).get("fallback_providers", [])
        if fallback_nodes:
            fb = fallback_nodes[0]
            return wrap([
                make_fallback(t, fb) if t["task_id"] == failing_id else t
                for t in current_plan["tasks"]
            ])

        # 3️⃣ remove failing task
        return wrap([t for t in current_plan["tasks"] if t["task_id"] != failing_id])

    # ----------------- LLM replanning mode -----------------

    replanner_prompt = """
You are a replanner. Never create placeholder tasks.

Rules:
1) Retry failed task if allowed (with relaxed inputs)
2) Else use fallback provider if available
3) Else remove the failing task and rebuild a minimal valid plan

Return ONLY valid JSON.
"""

    try:
        chain_prompt = ChatPromptTemplate.from_messages([
            ("system", replanner_prompt),
            ("user", "CURRENT_PLAN: {plan}\nFAILURE: {failure}\nMANIFEST: {manifest}")
        ])

        result = safe_invoke_planner(
            chain_prompt,
            {
                "plan": current_plan,
                "failure": failure_context,
                "manifest": manifest,
            }
        )

        if not result.get("ok"):
            raise ValueError("LLM replanner failed")

        plan = result.get("plan")

        # minimal schema validation (same behavior, safer)
        if (
            not isinstance(plan, dict)
            or "tasks" not in plan
            or not isinstance(plan["tasks"], list)
        ):
            raise ValueError("Invalid LLM plan schema")

        return plan

    except Exception:
        # fallback to deterministic logic (same as no-LLM mode)
        failed_task = find_task(current_plan.get("tasks", []), failing_id)
        if not failed_task:
            return wrap(current_plan.get("tasks", []))

        node = failed_task["node"]

        max_retries = manifest.get(node, {}).get("retry_strategy", {}).get("max_retries", 0)
        if max_retries > 0:
            return wrap([
                make_retry(t) if t["task_id"] == failing_id else t
                for t in current_plan["tasks"]
            ])

        fallback_nodes = manifest.get(node, {}).get("fallback_providers", [])
        if fallback_nodes:
            fb = fallback_nodes[0]
            return wrap([
                make_fallback(t, fb) if t["task_id"] == failing_id else t
                for t in current_plan["tasks"]
            ])

        return wrap([t for t in current_plan["tasks"] if t["task_id"] != failing_id])
