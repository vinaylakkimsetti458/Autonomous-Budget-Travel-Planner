import streamlit as st
from datetime import datetime, timedelta

from langchain_core.messages import SystemMessage
from main import create_and_run_planner

from agents.email_agent import (
    send_trip_plan_email,
    generate_financial_df,
    style_financial_df,
    
)

from utils.helpers import (
    calculate_duration,
    CITY_IATA_MAP,
    SENDER_EMAIL,
    SENDER_PASSWORD,
    LLM_READY,
)

from apis.amadeus_api import AMADEUS_CLIENT_READY

def main():
    st.set_page_config(page_icon="🌍", layout="wide", page_title="Multi-Agent Budget Travel Planner")
    st.title("🌍 Autonomous Budget-Aware Travel Planning System")
    st.markdown("Use the sidebar to input your trip details, and let the multi-agent system calculate costs and generate a custom itinerary based on your budget.")
    st.caption(f"**Backend Status:** LLM: {'✅ Ready' if LLM_READY else '❌ Disabled (Mocking)'} | Amadeus API: {'✅ Connected' if AMADEUS_CLIENT_READY else '❌ Mocking'}")

    if 'run_successful' not in st.session_state:
        st.session_state['run_successful'] = False
        st.session_state['final_state'] = None
        st.session_state['budget_review'] = None
        st.session_state['email_status'] = None
        st.session_state['recipient_email'] = ""

    with st.sidebar:
        st.header("1. Plan Your Trip")
        city_label = st.selectbox("Destination City (Origin is Hyderabad, India)", list(CITY_IATA_MAP.keys()), index=list(CITY_IATA_MAP.keys()).index("Aalborg, Denmark(AAL)") if "Aalborg, Denmark(AAL)" in CITY_IATA_MAP else 0)
        selected_city_name = city_label.split(',')[0].strip()
        if 'New York' in city_label:
            selected_city_name = 'New York'
        selected_iata = CITY_IATA_MAP[city_label]
        today = datetime.now().date()
        default_start_date = today + timedelta(days=30)
        default_end_date = default_start_date + timedelta(days=4)
        start_date = st.date_input("Start Date", default_start_date)
        end_date = st.date_input("End Date", default_end_date)
        budget = st.number_input("Total Budget (in ₹ Indian Rupees)", min_value=10000.00, value=200000.00, step=5000.00, format="%.2f")
        st.markdown("---")
        if st.button("Generate Trip Plan ✈️", type="primary", use_container_width=True):
            if start_date >= end_date:
                st.error("End Date must be after the Start Date.")
            else:
                duration = calculate_duration(start_date, end_date)
                initial_state = {
                    "city": selected_city_name,
                    "city_iata": selected_iata,
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d"),
                    "budget": budget,
                    "duration_days": duration,
                    "messages": [SystemMessage(content=f"Planning trip to {selected_city_name} for {duration} days.")],
                    "remaining_budget": budget,
                    "flight_cost": 0.0, "accommodation_cost": 0.0, "food_cost": 0.0, "activities_cost": 0.0,
                    "flight_details": "", "accommodation_details": "", "food_itinerary": [], "activities_plan": [], "itinerary_draft": "", "itinerary_extras": "",
                    "is_budget_met": False, "next_action": "FLIGHT_AGENT"
                }
                with st.spinner("🚀 Running Multi-Agent Planning Graph..."):
                    final_state = create_and_run_planner(initial_state)
                    
                    if final_state.get("remaining_budget", 0.0) < 0:
                        st.session_state["budget_review"] = {
                            "suggestion": final_state.get("suggestion"),
                            "action": final_state.get("action"),
                        }

                st.session_state['run_successful'] = True
                st.session_state['final_state'] = final_state
                st.session_state['email_status'] = None
                st.rerun()

    if st.session_state['run_successful']:
        final_state = st.session_state['final_state']
        st.header(f"Trip Summary: {final_state.get('city', 'Unknown')}")
        st.subheader("💰 Budget Breakdown")
        st.dataframe(style_financial_df(generate_financial_df(final_state)), use_container_width=True, hide_index=True)

        if final_state.get('remaining_budget', 0.0) < 0 and 'budget_review' in st.session_state:
            st.error("Budget Exceeded! See Review Below.")
            review_data = st.session_state['budget_review']
            st.subheader("🎯 Budget Review Suggestions")
            st.markdown(review_data['suggestion'])
            st.warning(review_data['action'])
        elif final_state.get('remaining_budget', 0.0) >= 0:
            st.success(f"✅ Plan is within budget! Remaining: ₹{final_state.get('remaining_budget', 0.0):,.2f}")

        st.subheader("✈️ Flight & Hotel Details")
        flight_details_str = str(final_state.get('flight_details', ''))
        if "|" in flight_details_str:
            outbound_str, return_str = flight_details_str.split("|", 1)
            col_out, col_ret = st.columns(2)
            with col_out:
                st.info(f"**🛫 Outbound Flight**\n\n{outbound_str.strip()}")
            with col_ret:
                st.info(f"**🛬 Return Flight**\n\n{return_str.strip()}")
        else:
            st.info(f"**Flight (Total Round Trip):** {flight_details_str}")
        st.info(f"**🏨 Hotel:** {final_state.get('accommodation_details', '')}")
        st.markdown("---")

        st.header("🗺️ Generated Itinerary")
# show the LLM-produced itinerary draft (always prefer this)
        it_text = final_state.get('itinerary_draft', '')
        if not it_text:
            st.info("Itinerary not available. This may happen if the LLM failed — check logs.")
        else:
            # Render the itinerary markdown (keep it friendly)
            st.markdown(it_text)

        # Debug trace (kept, but separate)
        with st.expander("🔍 Debug Trace (agent outputs)"):
            trace = final_state.get("trace", [])
            st.json(trace)

        # Note: the raw LLM output is saved to session_state['last_raw_itinerary'] for debugging,
        # but we do NOT display it by default to the end-user (to avoid confusion).
        st.markdown("---")
        st.subheader("📬 Send Plan via Email")
    
        recipient_email_input = st.text_input("Enter the email to send the plan:", value=st.session_state['recipient_email'], key="email_input")
        st.session_state['recipient_email'] = recipient_email_input

        if st.button("Email Me the Full Plan 📧", type="secondary"):
            if not st.session_state['recipient_email'] or "@" not in st.session_state['recipient_email']:
                st.error("Please enter a valid recipient email address.")
            elif not SENDER_EMAIL or not SENDER_PASSWORD:
                st.error("Email credentials are missing. Check your .env file for EMAIL_ADDRESS and EMAIL_PASSWORD.")
            else:
                email_to_send = st.session_state['recipient_email']
                review_data_to_send = st.session_state.get('budget_review') if final_state.get('remaining_budget', 0.0) < 0 else None
                with st.spinner(f"Sending plan to {email_to_send}..."):
                    success = send_trip_plan_email(final_state, email_to_send, SENDER_EMAIL, SENDER_PASSWORD, generate_financial_df, review_data_to_send)
                    st.session_state['email_status'] = (success, email_to_send)
                    st.rerun()

        if st.session_state['email_status']:
            success, email = st.session_state['email_status']
            if success:
                st.success(f"✅ Trip plan successfully sent to **{email}**!")
            else:
                st.error("❌ Failed to send email. Check console for SMTP error details.")

if __name__ == '__main__':
    main()