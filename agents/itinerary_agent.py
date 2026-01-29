import streamlit as st
from typing import Optional, Dict, Any
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from models.planner_state import PlannerState
from utils.helpers import GROQ_LLM, invoke_llm_with_timeout

def itinerary_planner_agent(state: PlannerState) -> Dict[str, Any]:
    """
    Day-by-day itinerary generator. Calls the LLM once per day (safer than one huge call).
    Returns the same dict shape as before:
      {"itinerary_draft": str, "messages": [...], "next_action": "FINISH"}
    Stores raw outputs for debugging in st.session_state['last_raw_itinerary'] (concatenated).
    """
    STOP_PHRASE = "---END_OF_ITINERARY---"
    city = state.get("city", "City")
    duration = int(state.get("duration_days", 1) or 1)
    remaining_budget = state.get("remaining_budget", 0.0)
    cache_key = f"itinerary__{city}_{duration}_{int(remaining_budget)}"
    if st.session_state.get(cache_key):
        return {
            "itinerary_draft": st.session_state[cache_key],
            "messages": [SystemMessage(content="Itinerary served from cache.")],
            "next_action": "FINISH"
        }

    # Helper to extract activity lines relevant to a day from activities_plan list
    activities_lines = state.get("activities_plan", []) or []
    activities_by_day = {d: [] for d in range(1, duration + 1)}
    for line in activities_lines:
        # Expected forms like "Day 2 Morning: Name (₹...)" or "Day 1 Morning: ..."
        try:
            if line.lower().startswith("day "):
                parts = line.split(":", 1)
                left = parts[0]  # "Day X Morning"
                day_token = left.strip().split()[1]
                day_idx = int(day_token)
                activities_by_day.setdefault(day_idx, []).append(line)
        except Exception:
            continue

    # Build a compact per-day prompt template (less brittle than huge rule dump)
    per_day_prompt_template = """You are a professional travel writer. Produce only the **Day {day}** block in EXACT markdown headings below for {city}.

## Day {day}

### 🌅 Morning
Create EXACTLY 4 short bullets (max 18–20 words each).  
Bullet 1 MUST follow this pattern EXACTLY:

- Have breakfast at **<RESTAURANT>**and then visit ***<ACTIVITY NAME> (₹<PRICE>)***.

Bullets 2–4:
- Short movement / cultural / transport bullets  
- DO NOT repeat restaurant or activity names  


### 🌤 Afternoon
Same rules as Morning:  
4 bullets, Bullet 1 MUST include:
- Lunch restaurant in **bold** 
- One activity in ***bold italics*** with price


### 🌙 Evening
Same rules as above:  
4 bullets including:
- Dinner restaurant (**bold**)  
- Evening activity (***bold italics*** with price) 
- 2–3 short movement bullets
- Don't add Google Map links for : 🧭 Logistics, 🌦 Weather Notes, 🎒 Packing Tips, 📸 Photo Spots, 🚇 Transport Guide, 🔐 Safety Tips, 💸 Budget Alternatives.

### MARKDOWN FORMATTING RULES (STRICT)
- Restaurants = **bold**  
- Activities = ***bold italics*** with price  
- Headings must be:
  ## Day {day}
  ### 🌅 Morning
  ### 🌤 Afternoon
  ### 🌙 Evening
- No extra commentary outside the Day block

### Additional Sections (3 bullets each)
After Evening, add:

### 🧭 Logistics
### 🌦 Weather Notes
### 🎒 Packing Tips
### 📸 Photo Spots
### 🚇 Transport Guide
### 🔐 Safety Tips
### 💸 Budget Alternatives

Each section MUST have exactly 3 short bullets.

### INPUTS
City: {city}  
Day index: {day}  
Activities you may use:  
{activities}

Output ONLY the markdown for **Day {day}**, nothing else.

End the output with {stop_phrase} (optional).
"""

    # If LLM disabled, produce a short fallback for every day
    if not GROQ_LLM:
        combined = ""
        for d in range(1, duration + 1):
            combined += f"## Day {d}: Short fallback for {city}\n"
            combined += "- 🌅 Morning: Arrive / local transport.\n- 🌤 Afternoon: Basic sightseeing.\n- 🌙 Evening: Dinner & relax.\n\n"
        st.session_state[cache_key] = combined
        st.session_state['last_raw_itinerary'] = "(LLM Disabled)"
        return {"itinerary_draft": combined, "messages": [SystemMessage(content="Itinerary (fallback) completed.")], "next_action": "FINISH"}

    # invoke helper (reuses invoke_llm_with_timeout)
    def call_llm_for_day(day_idx: int, attempt_timeout: float = 40.0, attempts: int = 2) -> str:
        activities_text = "\\n".join(activities_by_day.get(day_idx, [])) or "(no specific activities provided)"
        prompt_text = per_day_prompt_template.format(day=day_idx, city=city, activities=activities_text, stop_phrase=STOP_PHRASE)
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_text),
            ("user", "Generate Day {day} block now.".format(day=day_idx))
        ])
        chain = prompt | GROQ_LLM
        last_raw = ""
        for a in range(1, attempts + 1):
            ok, raw = invoke_llm_with_timeout(chain, {}, timeout=attempt_timeout)
            if ok and raw:
                raw_txt = (raw or "").strip()
                
                last_raw = raw_txt
                # If STOP_PHRASE present, cut at it
                if STOP_PHRASE in raw_txt:
                    return raw_txt.split(STOP_PHRASE)[0].strip()
                return raw_txt
            else:
                # record raw or error
                last_raw = raw or (last_raw or "")
                # small backoff (non-blocking here)
                # try again unless this was the last attempt
                continue
        # store the last_raw for diagnostics (kept externally)
        return ""  # indicate failure for this day

    combined_blocks = []
    raw_collector = []
    # iterate days and call LLM for each
    for d in range(1, duration + 1):
        with st.spinner(f"🗓 Generating Day {d}..."):
            day_text = call_llm_for_day(d, attempt_timeout=50.0, attempts=2)
            if day_text:
                # Ensure heading starts with '## Day d' — if model omitted heading, add it
                if not day_text.strip().startswith(f"## Day {d}"):
                    day_text = f"## Day {d}\n" + day_text
                combined_blocks.append(day_text.strip())
                raw_collector.append(day_text.strip())
            else:
                # Per-day fallback: short deterministic block (keeps format)
                fallback_block = (
                    f"## Day {d}: {city}\n"
                    f"### 🌅 Morning\n"
                    f"- Have breakfast at **Local Cafe** and then visit ***City Walk (₹0)***. [Google Maps](https://maps.google.com/?q={city}%20Local%20Cafe) | [Google Maps](https://maps.google.com/?q={city}%20City%20Walk)\n"
                    f"- Walk 1.2 km (15 min) to the old quarter.\n"
                    f"- Explore neighbourhood streets and markets.\n"
                    f"- Short break at a viewpoint.\n\n"
                    f"### 🌤 Afternoon\n"
                    f"- Have lunch at **Central Eatery** and then visit ***City Museum (₹300)***. [Google Maps](https://maps.google.com/?q={city}%20Central%20Eatery) | [Google Maps](https://maps.google.com/?q={city}%20City%20Museum)\n"
                    f"- 2 km bus (15 min).\n"
                    f"- Visit main plaza and gallery.\n"
                    f"- Coffee at a local spot.\n\n"
                    f"### 🌙 Evening\n"
                    f"- Have dinner at **Harbour Kitchen** and then join ***Sunset Promenade (₹0)***. [Google Maps](https://maps.google.com/?q={city}%20Harbour%20Kitchen) | [Google Maps](https://maps.google.com/?q={city}%20Sunset%20Promenade)\n"
                    f"- Walk back along waterfront (1.5 km, 20 min).\n"
                    f"- Enjoy local nightlife briefly.\n"
                    f"- Return to hotel and rest.\n\n"
                    f"### 🧭 Logistics\n- Accommodation: {state.get('accommodation_details','N/A')}\n- Transit: local buses and taxis available\n- Meeting point: hotel lobby\n\n"
                    f"### 🌦 Weather Notes\n- Typical temps and packing notes for the city.\n- Check forecast day-of\n- Bring compact umbrella\n\n"
                    f"### 🎒 Packing Tips\n- Comfortable shoes\n- Lightweight layers\n- Power adapter\n\n"
                    f"### 📸 Photo Spots\n- City skyline, Main plaza, Waterfront\n\n"
                    f"### 🚇 Transport Guide\n- Typical short transit suggestions\n\n"
                    f"### 🔐 Safety Tips\n- Keep valuables secure\n- Be cautious of crowded areas\n\n"
                    f"### 💸 Budget Alternatives\n- Choose street food, free parks, free walking tours\n"
                )
                combined_blocks.append(fallback_block)
                raw_collector.append(f"(FALLBACK_DAY_{d})")

    full_itinerary = "\n\n".join(combined_blocks).strip()
    # store a combined version of raw outputs for debugging
    st.session_state['last_raw_itinerary'] = ("\n\n--- RAW DAY OUTPUTS ---\n\n".join(raw_collector))[:200000]

    # final caching and return
    st.session_state[cache_key] = full_itinerary
    return {
        "itinerary_draft": full_itinerary,
        "messages": [SystemMessage(content="Itinerary draft completed (day-by-day).")],
        "next_action": "FINISH"
    }