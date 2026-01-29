import smtplib
import pandas as pd
from typing import Dict, Any, Callable, Optional
from email.mime.text import MIMEText
from models.planner_state import PlannerState

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

def generate_financial_df(final_state: PlannerState) -> pd.DataFrame:
    total_spent = (final_state.get('flight_cost', 0.0) + final_state.get('accommodation_cost', 0.0) + final_state.get('food_cost', 0.0) + final_state.get('activities_cost', 0.0))
    activities_plan = final_state.get("activities_plan", [])
    if activities_plan:
        max_items_to_show = 4
        preview_items = activities_plan[:max_items_to_show]
        activities_details = " | ".join(preview_items)
        if len(activities_plan) > max_items_to_show:
            activities_details += f" ... (+{len(activities_plan) - max_items_to_show} more)"
    else:
        activities_details = "Amadeus activities & local experiences"
    data = {
        'Category': ['Total Budget (Initial)', 'Flight', 'Accommodation', 'Food & Dining', 'Activities & Local Travel', 'Total Spent', 'Remaining Budget'],
        'Details': ['Original Goal', final_state.get('flight_details', ''), final_state.get('accommodation_details', ''), f"{final_state.get('duration_days', 0)} Days of meals", activities_details, '', ''],
        'Cost (₹)': [final_state.get('budget', 0.0), final_state.get('flight_cost', 0.0), final_state.get('accommodation_cost', 0.0), final_state.get('food_cost', 0.0), final_state.get('activities_cost', 0.0), total_spent, final_state.get('remaining_budget', 0.0)]
    }
    df = pd.DataFrame(data)
    return df

def style_financial_df(df: pd.DataFrame):
    return df.style.format({'Cost (₹)': '₹{:,.2f}'})

def send_trip_plan_email(final_state: Dict[str, Any], recipient: str, sender_email: str, sender_password: str, generate_financial_df: Callable, budget_review_data: Optional[Dict[str, str]] = None) -> bool:
    if not all([sender_email, sender_password]):
        print("ERROR: Email credentials (sender email or sender_password) not found in the function call.")
        return False
    try:
        budget_df = generate_financial_df(final_state)
    except Exception as e:
        print(f"ERROR building budget DataFrame: {e}")
        budget_df = None
    budget_summary_text = ""
    if budget_df is not None:
        try:
            if isinstance(budget_df, pd.DataFrame):
                budget_summary_text = budget_df.to_string(index=False)
            elif hasattr(budget_df, "to_string"):
                budget_summary_text = budget_df.to_string(index=False)
            else:
                budget_summary_text = str(budget_df)
        except Exception as e:
            print(f"WARNING: Failed to stringify budget_df: {e}")
            try:
                budget_summary_text = str(budget_df)
            except Exception:
                budget_summary_text = "Budget summary unavailable."
    else:
        budget_summary_text = "Budget summary unavailable."

    budget_review_section = ""
    if budget_review_data:
        budget_review_section = f"""
======================================================================
🚨 BUDGET REVIEW SUGGESTIONS 🚨
======================================================================
**Suggestion:** {budget_review_data.get('suggestion', 'N/A')}
**Action:** {budget_review_data.get('action', 'N/A')}
----------------------------------------------------------------------
"""

    city = final_state.get('city', 'Unknown destination')
    remaining_budget_val = final_state.get('remaining_budget', 0.0)
    try:
        remaining_str = f"₹{float(remaining_budget_val):,.2f}"
    except Exception:
        remaining_str = str(remaining_budget_val)
    start_date = final_state.get('start_date', 'N/A')
    end_date = final_state.get('end_date', 'N/A')
    flight_cost_val = final_state.get('flight_cost', 0.0)
    try:
        flight_cost_str = f"₹{float(flight_cost_val):,.2f}"
    except Exception:
        flight_cost_str = str(flight_cost_val)
    raw_flight_details = final_state.get('flight_details', '')
    flight_details_str = str(raw_flight_details).replace('|', '\n* ') if raw_flight_details is not None else "No flight details available."
    itinerary_text = final_state.get('itinerary_draft', '')
    itinerary_text = str(itinerary_text) if itinerary_text is not None else "(no itinerary provided)"

    email_body = f"""
Trip Summary: {city}
----------------------------------------------------------------------
Budget Status: {'✅ Plan is within budget!' if remaining_budget_val >= 0 else '❌ Budget Exceeded!'} Remaining: {remaining_str}
Trip Dates: {start_date} to {end_date}

======================================================================
💰 BUDGET BREAKDOWN
======================================================================
{budget_summary_text}
{budget_review_section}

----------------------------------------------------------------------
✈️ FLIGHT & HOTEL DETAILS
----------------------------------------------------------------------
* **Total Round Trip Flight Cost:** {flight_cost_str}
* **Flight Details:** {flight_details_str}
* **Hotel:** {final_state.get('accommodation_details', 'N/A')}

======================================================================
🗺️ GENERATED ITINERARY
======================================================================
{itinerary_text}
"""
    try:
        msg = MIMEText(email_body, 'plain', 'utf-8')
        msg['Subject'] = f"Trip Plan: {city} ({start_date} - {end_date})"
        msg['From'] = sender_email
        msg['To'] = recipient
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, [recipient], msg.as_string())
            return True
    except Exception as e:
        print(f"SMTP Error Details: {e}")
        return False

