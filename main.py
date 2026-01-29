import traceback
from typing import Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END

from models.planner_state import PlannerState
from utils.helpers import MAX_REPLANS, CAPABILITY_MANIFEST

from agents.planner_agent import planner_agent
from agents.replanner_agent import replanner_agent
from agents.total_cost_agent import total_cost_check

from agents.flight_agent import flight_agent
from agents.accommodation_agent import accommodation_agent
from agents.food_agent import food_agent
from agents.activities_agent import activities_agent
from agents.budget_review_agent import budget_review_agent
from agents.itinerary_agent import itinerary_planner_agent

MAX_WORKERS = 3     

def route_after_costing(state: PlannerState) -> str:
    if state.get('remaining_budget', 0.0) < 0:
        return "BUDGET_REVIEW"
    else:
        return "ITINERARY_PLANNER"

def _run_legacy_graph(user_initial_state: PlannerState):
    workflow = StateGraph(PlannerState)
    workflow.add_node("FLIGHT_AGENT", lambda s: flight_agent(s))
    workflow.add_node("ACCOMMODATION_AGENT", lambda s: accommodation_agent(s))
    workflow.add_node("FOOD_AGENT", lambda s: food_agent(s))
    workflow.add_node("ACTIVITIES_AGENT", lambda s: activities_agent(s))
    workflow.add_node("TOTAL_COST_CHECK", lambda s: total_cost_check(s))
    workflow.add_node("BUDGET_REVIEW", lambda s: budget_review_agent(s))
    workflow.add_node("ITINERARY_PLANNER", lambda s: itinerary_planner_agent(s))
    workflow.set_entry_point("FLIGHT_AGENT")
    workflow.add_edge("FLIGHT_AGENT", "ACCOMMODATION_AGENT")
    workflow.add_edge("ACCOMMODATION_AGENT", "FOOD_AGENT")
    workflow.add_edge("FOOD_AGENT", "ACTIVITIES_AGENT")
    workflow.add_edge("ACTIVITIES_AGENT", "TOTAL_COST_CHECK")
    workflow.add_conditional_edges("TOTAL_COST_CHECK", route_after_costing, {"BUDGET_REVIEW": "BUDGET_REVIEW", "ITINERARY_PLANNER": "ITINERARY_PLANNER"})
    workflow.add_edge("BUDGET_REVIEW", "ITINERARY_PLANNER")
    workflow.add_edge("ITINERARY_PLANNER", END)
    app = workflow.compile()
    final_state = app.invoke(user_initial_state)
    return final_state

def create_and_run_planner(user_initial_state: PlannerState):
    """
    Autonomous orchestrator (flight synchronous, other cost agents parallel).
    - Flight runs synchronously first (if present at plan start).
    - Accommodation/Food/Activities run concurrently (ThreadPoolExecutor).
    - Accommodation/Food/Activities must run parallely as not always Accomodation last.
    - Not always run Accomodation Agent last when Accommodation/Food/Activities agents are running in parallel.
    - Worker threads return partial outputs; main thread merges them.
    - Replanner is called if any worker fails.
    """
    def merge_partial_state(main_state: Dict[str, Any], partial: Optional[Dict[str, Any]]):
        """Merge an agent's partial output into main_state robustly."""
        if not isinstance(partial, dict):
            return
        for k, v in partial.items():
            if k == "messages":
                main_state.setdefault("messages", [])
                if isinstance(v, list):
                    main_state["messages"].extend(v)
                else:
                    main_state["messages"].append(v)
            elif k == "trace":
                main_state.setdefault("trace", [])
                if isinstance(v, list):
                    main_state["trace"].extend(v)
                else:
                    main_state["trace"].append(v)
            else:
                # Overwrite other keys (costs, details, next_action, remaining_budget...)
                main_state[k] = v

    # Build initial user_goal/context for planner
    user_goal = {
        "city": user_initial_state["city"],
        "city_iata": user_initial_state["city_iata"],
        "start_date": user_initial_state["start_date"],
        "end_date": user_initial_state["end_date"],
        "duration_days": user_initial_state["duration_days"],
        "budget": user_initial_state["budget"],
    }
    context = {"messages": [m.content for m in user_initial_state.get("messages", [])]}

    plan = planner_agent(user_goal, CAPABILITY_MANIFEST, context)
    try:
        import json
        print("PLANNER PLAN:", json.dumps(plan, indent=2))
    except Exception:
        print("PLANNER PLAN (unprintable)")

    tasks = plan.get("tasks", []) or []
    if not tasks:
        # fallback to legacy if planner returned nothing
        return _run_legacy_graph(user_initial_state)

    # Mutable runtime state
    state = dict(user_initial_state)
    for k in ["flight_cost", "accommodation_cost", "food_cost", "activities_cost", "remaining_budget"]:
        state.setdefault(k, 0.0)
    state.setdefault("messages", state.get("messages", []))
    state.setdefault("trace", state.get("trace", []))

    NODE_TO_FUNC = {
        "FLIGHT_AGENT": flight_agent,
        "ACCOMMODATION_AGENT": accommodation_agent,
        "FOOD_AGENT": food_agent,
        "ACTIVITIES_AGENT": activities_agent,
        "TOTAL_COST_CHECK": total_cost_check,
        "BUDGET_REVIEW": budget_review_agent,
        "ITINERARY_PLANNER": itinerary_planner_agent
    }

    # Helper: run an agent in worker safely (returns partial output dict)
    def run_agent_worker(node: str, agent_fn: Callable, local_state_snapshot: Dict[str, Any], inputs: Dict[str, Any], attempt: int = 1):
        """
        Worker wrapper: receives a shallow copy of state (read-only), calls agent_fn,
        returns tuple (task_id, success(bool), partial_output_or_error_dict).
        """
        task_id = inputs.get("task_id") or f"{node}"
        try:
            # Agent might expect 'task_inputs' or not; try both
            try:
                out = agent_fn(local_state_snapshot, task_inputs=inputs)
            except TypeError:
                out = agent_fn(local_state_snapshot)
            if not isinstance(out, dict):
                out = {"messages": [SystemMessage(content=f"Agent {node} returned non-dict output")] }
            return task_id, True, out
        except Exception as e:
            # Build a failure partial with trace for merging (main thread will see it)
            tb = traceback.format_exc()
            err_partial = {
                "messages": [SystemMessage(content=f"Agent {node} failed on attempt {attempt}: {e}")],
                "trace": [{"node": node, "error": str(e), "traceback": tb}]
            }
            return task_id, False, err_partial

    # ---------- Execution logic ----------
    i = 0
    replan_count = 0

    # If first task is flight, execute it synchronously before parallelizing others
    if tasks and tasks[0].get("node") == "FLIGHT_AGENT":
        flight_task = tasks[0]
        try:
            out = NODE_TO_FUNC["FLIGHT_AGENT"](state, task_inputs=flight_task.get("inputs", {}) or {})
        except TypeError:
            out = NODE_TO_FUNC["FLIGHT_AGENT"](state)
        except Exception as e:
            out = {"messages": [SystemMessage(content=f"Flight agent failed: {e}")], "trace": [{"node":"FLIGHT_AGENT","error":str(e)}]}
        merge_partial_state(state, out)

        # recompute remaining_budget
        try:
            state["remaining_budget"] = state.get("budget", 0.0) - sum([
                float(state.get("flight_cost", 0.0)),
                float(state.get("accommodation_cost", 0.0)),
                float(state.get("food_cost", 0.0)),
                float(state.get("activities_cost", 0.0)),
            ])
        except Exception:
            pass

        i = 1  # move past flight
    else:
        i = 0

    # Process rest of tasks: parallel for contiguous 'parallel' tasks, else sync
    while i < len(tasks):
        # Build a batch (contiguous parallel group or single non-parallel)
        batch = [tasks[i]]
        if tasks[i].get("parallel"):
            j = i + 1
            while j < len(tasks) and tasks[j].get("parallel"):
                batch.append(tasks[j])
                j += 1
            next_index = j
        else:
            next_index = i + 1

        # If batch contains more than 1 agent, run them concurrently
        if len(batch) > 1:
            # Submit all workers
            futures = {}
            results = {}
            failures = []
            with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(batch))) as ex:
                for t in batch:
                    node = t.get("node")
                    agent_fn = NODE_TO_FUNC.get(node)
                    # pass a shallow copy of state to avoid mutation inside worker
                    state_snapshot = dict(state)
                    task_inputs = t.get("inputs", {}) or {}
                    # include a task_id inside inputs for traceability
                    task_inputs = dict(task_inputs)
                    task_inputs["task_id"] = t.get("task_id", f"{node}")
                    fut = ex.submit(run_agent_worker, node, agent_fn, state_snapshot, task_inputs, 1)
                    futures[fut] = t

                # Collect results as they complete
                for fut in as_completed(futures):
                    t = futures[fut]
                    node = t.get("node")
                    try:
                        task_id, success, out = fut.result()
                    except Exception as e:
                        task_id = t.get("task_id", f"{node}")
                        success = False
                        tb = traceback.format_exc()
                        out = {"messages":[SystemMessage(content=f"Worker crashed for {node}: {e}")], "trace":[{"node": node, "error": str(e), "traceback": tb}]}
                    results[task_id] = (success, out)
                    if not success:
                        failures.append(task_id)

            # Merge all results in main thread
            for tid, (ok, partial) in results.items():
                merge_partial_state(state, partial)

            # Recompute remaining budget after merging the parallel outputs
            try:
                state["remaining_budget"] = state.get("budget", 0.0) - sum([
                    float(state.get("flight_cost", 0.0)),
                    float(state.get("accommodation_cost", 0.0)),
                    float(state.get("food_cost", 0.0)),
                    float(state.get("activities_cost", 0.0)),
                ])
            except Exception:
                pass

            # If any failures occurred, call replanner and restart execution
            if failures:
                failure_context = {
                        "failing_task_id": failures[0],  # pick first failed task
                        "failed_tasks": failures,
                        "state_snapshot": state
                    }
                print(f"Parallel batch failures: {failure_context}")
                plan = replanner_agent(failure_context, CAPABILITY_MANIFEST, plan)
                tasks = plan.get("tasks", [])
                replan_count += 1
                if replan_count > MAX_REPLANS:
                    print("Max replans exceeded: applying budget review and finishing.")
                    merge_partial_state(state, budget_review_agent(state))
                    merge_partial_state(state, itinerary_planner_agent(state))
                    return state
                # restart plan execution from beginning (flight might or might not be present)
                i = 0
                continue

            # success -> advance index
            i = next_index
            continue

        # Single (non-parallel) task: execute synchronously
        else:
            task = batch[0]
            node = task.get("node")
            task_id = task.get("task_id", f"t{i+1}")
            agent_fn = NODE_TO_FUNC.get(node)
            if not agent_fn:
                print(f"Unknown node {node}, skipping.")
                i = next_index
                continue

            task_inputs = task.get("inputs", {}) or {}
            try:
                try:
                    out = agent_fn(state, task_inputs=task_inputs)
                except TypeError:
                    out = agent_fn(state)
            except Exception as e:
                tb = traceback.format_exc()
                out = {"messages":[SystemMessage(content=f"Agent {node} failed: {e}")], "trace":[{"node": node, "error": str(e), "traceback": tb}]}

            merge_partial_state(state, out)

            # recompute remaining budget after each synchronous task
            try:
                state["remaining_budget"] = state.get("budget", 0.0) - sum([
                    float(state.get("flight_cost", 0.0)),
                    float(state.get("accommodation_cost", 0.0)),
                    float(state.get("food_cost", 0.0)),
                    float(state.get("activities_cost", 0.0)),
                ])
            except Exception:
                pass

            # Basic failure heuristics for synchronous node
            is_failure = False
            try:
                if node == "FLIGHT_AGENT" and float(state.get("flight_cost", 0.0)) <= 0:
                    is_failure = True
                if node == "ACCOMMODATION_AGENT" and float(state.get("accommodation_cost", 0.0)) <= 0:
                    is_failure = True
                if node == "FOOD_AGENT" and float(state.get("food_cost", 0.0)) <= 0:
                    is_failure = True
            except Exception:
                is_failure = True

            if is_failure:
                failure_context = {"failing_task_id": task_id, "node": node, "state_snapshot": state}
                print(f"Task {task_id}/{node} failed — calling replanner.")
                plan = replanner_agent(failure_context, CAPABILITY_MANIFEST, plan)
                tasks = plan.get("tasks", [])
                replan_count += 1
                if replan_count > MAX_REPLANS:
                    print("Max replans reached; aborting to BUDGET_REVIEW.")
                    merge_partial_state(state, budget_review_agent(state))
                    merge_partial_state(state, itinerary_planner_agent(state))
                    return state
                i = 0
                continue

            i = next_index
            continue

    # Final checks: if remaining budget negative, run budget review
    try:
        if float(state.get("remaining_budget", 0.0)) < 0:
            br_out = budget_review_agent(state)
            merge_partial_state(state, br_out)
    except Exception as e:
        print(f"Error during final budget review: {e}")

    # Final itinerary
    try:
        it_out = itinerary_planner_agent(state)
        merge_partial_state(state, it_out)
    except Exception as e:
        print(f"Itinerary planner agent failed at final stage: {e}")

    return state