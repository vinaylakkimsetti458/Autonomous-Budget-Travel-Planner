import os
import time
import json
import random
from datetime import datetime, date
from typing import Optional, Tuple, List, Dict, Any
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from utils.city_iata import CITY_IATA_MAP
from utils.city_bbox import CITY_BOUNDING_BOXES
from utils.fallback_names import FALLBACK_HOTEL_NAMES

load_dotenv()

EXCHANGE_RATES = {"USD": 89.45, "EUR": 103.52, "INR": 1.0}

MEAL_COST_TIERS = {"Low": 450.00, "Medium": 900.00, "High": 1600.00}

MAX_WORKERS = 6
MAX_RETRIES_PER_TASK = 2
BASE_RETRY_DELAY = 1.0  # seconds
MAX_REPLANS = 3

MAX_LLM_ATTEMPTS = 2
LLM_RETRY_BACKOFF = 1.0 


CITY_IATA_MAP = dict(sorted(CITY_IATA_MAP.items()))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SENDER_EMAIL = os.getenv("EMAIL_ADDRESS")
SENDER_PASSWORD = os.getenv("EMAIL_PASSWORD")

GROQ_LLM = None
LLM_READY = False

def init_groq_llm(api_key: Optional[str], model: str = "openai/gpt-oss-120b", max_tokens: int = 3000, timeout_sec: float = 30.0):
    """
    Try multiple common timeout arg names to avoid constructor errors across SDK versions.
    Returns (llm_instance_or_None, error_message_or_None)
    """
    if not api_key:
        return None, "GROQ_API_KEY missing"
    tried_exceptions = []
    kw_variants = [
        {"model": model, "temperature": 0.0, "api_key": api_key, "max_tokens": max_tokens, "request_timeout": timeout_sec},
        {"model": model, "temperature": 0.0, "api_key": api_key, "max_tokens": max_tokens, "timeout": timeout_sec},
        {"model": model, "temperature": 0.0, "api_key": api_key, "max_tokens": max_tokens, "REQUEST_TIMEOUT": timeout_sec},
        {"model": model, "temperature": 0.0, "api_key": api_key, "max_tokens": max_tokens}  # last-ditch (no timeout kw)
    ]
    for kw in kw_variants:
        try:
            llm = ChatGroq(**kw)
            return llm, None
        except Exception as e:
            tried_exceptions.append(f"Attempt with keys {list(kw.keys())} failed: {repr(e)}")
    return None, "\n".join(tried_exceptions)

GROQ_LLM, init_err = init_groq_llm(GROQ_API_KEY, model="meta-llama/llama-4-maverick-17b-128e-instruct", max_tokens=3000, timeout_sec=30.0)


if GROQ_LLM is None:
    print("GROQ LLM not initialized:", init_err)
    LLM_READY = False
else:
    print("GROQ LLM initialized OK.")
    LLM_READY = True

# ---------- Capability manifest ----------
CAPABILITY_MANIFEST = {
    "FLIGHT_AGENT": {"inputs": ["city_iata", "start_date", "end_date", "budget"], "outputs": ["flight_cost", "flight_details"], "parallelizable": False},
    "ACCOMMODATION_AGENT": {"inputs": ["city_iata", "start_date", "end_date", "duration_days", "budget"], "outputs": ["accommodation_cost", "accommodation_details"], "parallelizable": True},
    "FOOD_AGENT": {"inputs": ["city", "duration_days", "budget"], "outputs": ["food_cost", "food_itinerary"], "parallelizable": True},
    "ACTIVITIES_AGENT": {"inputs": ["city", "duration_days", "remaining_budget"], "outputs": ["activities_cost", "activities_plan"], "parallelizable": True},
    "TOTAL_COST_CHECK": {"inputs": [], "outputs": [], "parallelizable": False},
    "BUDGET_REVIEW": {"inputs": [], "outputs": [], "parallelizable": False},
    "ITINERARY_PLANNER": {"inputs": [], "outputs": ["itinerary_draft"], "parallelizable": False},
}

def get_city_bbox(city_name: str) -> Optional[str]:
    if not city_name:
        return None
    standardized_city_name = city_name.strip().title()
    if standardized_city_name in ["Newyork", "New York City"]:
        standardized_city_name = "New York"
    if standardized_city_name == "St Petersburg":
        standardized_city_name = "St Petersburg"
    return CITY_BOUNDING_BOXES.get(standardized_city_name, None)

def get_city_center_latlon(city_name: str) -> Optional[Tuple[float, float]]:
    bbox_str = get_city_bbox(city_name)
    if not bbox_str:
        return None
    try:
        parts = [p.strip() for p in bbox_str.split(",")]
        if len(parts) != 4:
            return None
        lon_min, lat_min, lon_max, lat_max = map(float, parts)
        center_lat = (lat_min + lat_max) / 2.0
        center_lon = (lon_min + lon_max) / 2.0
        return center_lat, center_lon
    except ValueError:
        return None
    
def convert_to_inr(price: float, currency_code: str) -> float:
    try:
        amount = float(price)
    except (TypeError, ValueError):
        return 0.0
    
    if not currency_code:
        currency_code = "INR"
    code = str(currency_code).strip().upper()
    EXCHANGE_RATES.setdefault("INR", 1.0)
    rate = EXCHANGE_RATES.get(code) or EXCHANGE_RATES.get(code[:3]) or 1.0

    return round(amount * rate, 2)

def calculate_duration(start_date_str: str, end_date_str: str) -> int:
    try:
        if isinstance(start_date_str, date):
            start_date_str = start_date_str.strftime("%Y-%m-%d")
        if isinstance(end_date_str, date):
            end_date_str = end_date_str.strftime("%Y-%m-%d")
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
        return (end_date - start_date).days + 1
    except Exception:
        return 3
    
def mock_activities_agent(city: str, duration_days: int, remaining_budget: float) -> Tuple[float, str]:
    estimated_daily_cost = random.uniform(3500,5000)
    initial_estimate = estimated_daily_cost * duration_days
    max_allowable_cost = max(0.0, remaining_budget * 0.3)
    cost = min(initial_estimate, max_allowable_cost, 0.25 * remaining_budget)
    cost = max(1500.0 * duration_days, cost)
    details = f"Estimated cost for entrance fees, local travel, and guided tours."
    return round(cost, 2), details

def build_daywise_activities_plan(activities: List[Dict[str, Any]], num_days: int, slots: Tuple[str, ...] = ("morning", "afternoon", "evening")) -> Dict[str, Dict[str, Optional[Dict[str, Any]]]]:
    plan: Dict[str, Dict[str, Optional[Dict[str, Any]]]] = {f"Day {d}": {slot: None for slot in slots} for d in range(1, num_days + 1)}
    if not activities:
        return plan
    priced = sorted([a for a in activities if a.get("price_inr") is not None], key=lambda x: x["price_inr"])
    total_slots = num_days * len(slots)
    selected = priced[:total_slots]
    for idx, act in enumerate(selected):
        day_index = idx // len(slots)
        slot_index = idx % len(slots)
        if day_index >= num_days:
            break
        day_name = f"Day {day_index + 1}"
        slot_name = slots[slot_index]
        plan[day_name][slot_name] = act
    return plan

def flatten_activities_plan_for_prompt(plan: Dict[str, Dict[str, Optional[Dict[str, Any]]]]) -> List[str]:
    lines: List[str] = []
    for day_name, slots in plan.items():
        for slot_name in ["morning", "afternoon", "evening"]:
            act = slots.get(slot_name)
            if not act:
                continue
            price = act.get("price_inr")
            if price is not None:
                lines.append(f"{day_name} {slot_name.title()}: {act['name']} (₹{price:,.0f})")
            else:
                lines.append(f"{day_name} {slot_name.title()}: {act['name']} (price not available)")
    return lines

def compute_total_used_activities_cost(plan: Dict[str, Dict[str, Optional[Dict[str, Any]]]]) -> float:
    seen = set()
    total = 0.0
    for day_name, slots in plan.items():
        for slot_name, act in slots.items():
            if not act:
                continue
            key = (act.get("name"), act.get("raw_amount"))
            if key in seen:
                continue
            seen.add(key)
            price = act.get("price_inr")
            if price is not None:
                total += price
    return round(total, 2)

def invoke_llm_with_timeout(chain_obj, inputs=None, timeout: float = 12.0):
    """
    Safely invoke an LLM chain/object with a timeout.
    - chain_obj: either a composed object (prompt | LLM) that supports .invoke(inputs)
                 or a callable that returns a response object/string.
    - inputs: dict or None
    - timeout: seconds to wait before giving up
    Returns: (ok: bool, raw_text_or_error: str)
    """
    inputs = inputs or {}
    def _call():
        # if chain_obj is callable that encapsulates .invoke, call it directly
        try:
            resp = chain_obj.invoke(inputs) if hasattr(chain_obj, "invoke") else chain_obj(inputs)
        except TypeError:
            # some chaining patterns might require no args
            resp = chain_obj.invoke() if hasattr(chain_obj, "invoke") else chain_obj()
        # normalize to string
        raw = getattr(resp, "content", None) or (resp if isinstance(resp, str) else str(resp))
        return raw

    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_call)
        try:
            raw = fut.result(timeout=timeout)
            return True, raw
        except FuturesTimeoutError:
            # Cancel and return timeout
            try:
                fut.cancel()
            except Exception:
                pass
            return False, f"timeout_after_{timeout}s"
        except Exception as e:
            return False, f"invoke_exception: {repr(e)}"
        
def safe_invoke_planner(chain_prompt, inputs, max_attempts=MAX_LLM_ATTEMPTS, backoff=LLM_RETRY_BACKOFF):
    """
    Call the LLM up to max_attempts times. Log raw output. Return dict:
      {"ok": True, "plan": <parsed JSON>} or {"ok": False, "error": "...", "last_raw": "..."}
    """
    last_raw = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = (chain_prompt | GROQ_LLM).invoke(inputs).content
            last_raw = resp
            # quick sanity parse
            parsed = json.loads(resp)
            return {"ok": True, "plan": parsed}
        except json.JSONDecodeError as jde:
            # Model returned non-JSON. Log raw output for debugging and return failure.
            print(f"[Planner] JSON parse error on attempt {attempt}: {jde}")
            print("=== RAW LLM RESPONSE (truncated, first 4000 chars) ===")
            print((last_raw or "")[:4000])
            print("=== END RAW LLM RESPONSE ===")
            return {"ok": False, "error": "invalid_json", "last_raw": (last_raw or "")[:4000]}
        except Exception as e:
            print(f"[Planner] LLM attempt {attempt} failed with exception: {repr(e)}")
            if last_raw:
                print("=== RAW LLM PARTIAL RESPONSE (truncated) ===")
                print(last_raw[:2000])
                print("=== END RAW PARTIAL ===")
            if attempt < max_attempts:
                time.sleep(backoff * attempt)
    return {"ok": False, "error": "llm_failed", "attempts": max_attempts, "last_raw": (last_raw[:2000] if last_raw else None)}


__all__ = [
    "EXCHANGE_RATES",
    "MEAL_COST_TIERS",
    "MAX_WORKERS",
    "MAX_RETRIES_PER_TASK",
    "BASE_RETRY_DELAY",
    "MAX_REPLANS",
    "CAPABILITY_MANIFEST",
    "GROQ_LLM",
    "LLM_READY",
    "get_city_bbox",
    "get_city_center_latlon",
    "convert_to_inr",
    "calculate_duration",
    "mock_activities_agent",
    "build_daywise_activities_plan",
    "flatten_activities_plan_for_prompt",
    "compute_total_used_activities_cost",
    "invoke_llm_with_timeout",
    "safe_invoke_planner",
]
