import os
import random
import requests
from dotenv import load_dotenv
load_dotenv()
from typing import Optional, Tuple, List, Dict, Any
from utils.fallback_names import (
    FALLBACK_HOTEL_NAMES,
    INDIAN_AIRLINES,
    INTERNATIONAL_AIRLINES
)
from utils.city_iata import INDIA_IATA_CODES
from utils.helpers import (
    convert_to_inr,
    get_city_center_latlon,
    mock_activities_agent,
    generate_best_flight,
    calculate_realistic_price
)

AUTH_URL = "https://test.api.amadeus.com/v1/security/oauth2/token"
FLIGHT_SEARCH_URL = "https://test.api.amadeus.com/v2/shopping/flight-offers"
HOTEL_SEARCH_URL = "https://test.api.amadeus.com/v1/reference-data/locations/hotels/by-city"
HOTEL_OFFERS_URL = "https://test.api.amadeus.com/v3/shopping/hotel-offers"
ACTIVITIES_URL = "https://test.api.amadeus.com/v1/shopping/activities"
REQUEST_TIMEOUT = 15

AMADEUS_CLIENT_ID = os.getenv("AMADEUS_CLIENT_ID", "TEST_ID")
AMADEUS_CLIENT_SECRET = os.getenv("AMADEUS_CLIENT_SECRET", "TEST_SECRET")

def is_international_city(city_iata: str) -> bool:
    return city_iata not in INDIA_IATA_CODES

def get_fallback_airline(city_iata: str) -> str:
    if is_international_city(city_iata):
        return random.choice(INTERNATIONAL_AIRLINES)
    return random.choice(INDIAN_AIRLINES)

def get_simulated_hotel_name(city_iata: str) -> str:
    return FALLBACK_HOTEL_NAMES.get(city_iata.upper(), FALLBACK_HOTEL_NAMES["DEFAULT"])

def get_amadeus_token(client_id: str, client_secret: str) -> Optional[str]:
    auth_data = {'grant_type': 'client_credentials', 'client_id': client_id, 'client_secret': client_secret}
    auth_headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    try:
        response = requests.post(AUTH_URL, headers=auth_headers, data=auth_data, timeout=5)
        response.raise_for_status()
        return response.json().get('access_token')
    except requests.exceptions.RequestException:
        return None

def initialize_amadeus() -> Tuple[bool, Optional[str]]:
    if "TEST" in AMADEUS_CLIENT_ID or "TEST" in AMADEUS_CLIENT_SECRET:
        return False, None
    token = get_amadeus_token(AMADEUS_CLIENT_ID, AMADEUS_CLIENT_SECRET)
    return bool(token), token

AMADEUS_CLIENT_READY, AMADEUS_TOKEN = initialize_amadeus()

def real_flight_api(city_iata: str, start_date: str, end_date: str) -> Tuple[float, str, str]:
    ORIGIN_IATA = "HYD"

    if not AMADEUS_CLIENT_READY or not AMADEUS_TOKEN:
        best = generate_best_flight(ORIGIN_IATA, city_iata)

        airline = best["airline"]
        stops = best["stops"]

        cost = calculate_realistic_price(city_iata, airline, stops)
        details = f"{airline} ({stops} stop(s))"

        return round(cost, 2), details, details

    search_headers = {'Authorization': f'Bearer {AMADEUS_TOKEN}'}
    search_params = {
        'originLocationCode': ORIGIN_IATA,
        'destinationLocationCode': city_iata,
        'departureDate': start_date,
        'returnDate': end_date,
        'adults': 1,
        'currencyCode': "INR",
        'max': 5
    }

    try:
        response = requests.get(
            FLIGHT_SEARCH_URL,
            headers=search_headers,
            params=search_params,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()
        flight_data = response.json()

        if not flight_data.get('data'):
            raise Exception("No flights")

        cheapest_offer = min(
            flight_data['data'],
            key=lambda offer: float(offer['price']['total'])
        )

        price = float(cheapest_offer['price']['total'])
        carrier_dict = flight_data.get('dictionaries', {}).get('carriers', {})

        outbound_segments = cheapest_offer['itineraries'][0]['segments']
        out_code = outbound_segments[0]['carrierCode']
        out_airline = carrier_dict.get(out_code, out_code)
        out_stops = len(outbound_segments) - 1

        outbound_details = f"{out_airline} ({out_stops} stop(s))"

        if len(cheapest_offer['itineraries']) > 1:
            return_segments = cheapest_offer['itineraries'][1]['segments']
            ret_code = return_segments[0]['carrierCode']
            ret_airline = carrier_dict.get(ret_code, ret_code)
            ret_stops = len(return_segments) - 1
        else:
            ret_airline = out_airline
            ret_stops = out_stops

        return_details = f"{ret_airline} ({ret_stops} stop(s))"

        return round(price, 2), outbound_details, return_details
    

    except Exception as e:
        print(f"⚠️ API failed → fallback: {e}")
    
        best = generate_best_flight(ORIGIN_IATA, city_iata)

        airline = best["airline"]
        stops = best["stops"]

        # smarter pricing
        cost = calculate_realistic_price(city_iata, airline, stops)

        details = f"{airline} ({stops} stop(s))"

        return round(cost, 2), details, details

    
def real_hotel_api(city_iata: str, start_date_str: str, end_date_str: str, duration_days: int) -> Tuple[float, str]:
    nights = duration_days - 1
    if nights <= 0:
        nights = 1

    simulated_name = get_simulated_hotel_name(city_iata)

    # If token missing → simulated return
    if not AMADEUS_CLIENT_READY or not AMADEUS_TOKEN:
        cost = random.uniform(15000, 20000) * nights
        return round(cost, 2), f"{simulated_name} for {nights} nights."

    # 1) Fetch hotel IDs
    headers = {'Authorization': f'Bearer {AMADEUS_TOKEN}'}
    try:
        res = requests.get(HOTEL_SEARCH_URL, headers=headers, params={'cityCode': city_iata}, timeout=REQUEST_TIMEOUT)
        res.raise_for_status()
        hotel_ids = [h.get("hotelId") for h in res.json().get("data", []) if h.get("hotelId")][:5]
        if not hotel_ids:
            raise Exception("No hotel IDs")
    except:
        cost = random.uniform(15000, 20000) * nights
        return round(cost, 2), f"{simulated_name} for {nights} nights."

    # 2) Fetch offers
    offers_params = {
        "hotelIds": ",".join(hotel_ids),
        "checkInDate": start_date_str,
        "checkOutDate": end_date_str,
        "currencyCode": "INR",
        "adults": 1
    }

    try:
        res = requests.get(HOTEL_OFFERS_URL, headers=headers, params=offers_params, timeout=REQUEST_TIMEOUT)
        res.raise_for_status()
        data = res.json().get("data", [])
        if not data:
            raise Exception("No offers")
    except:
        cost = random.uniform(15000, 20000) * nights
        return round(cost, 2), f"{simulated_name} for {nights} nights."

    # 3) Extract the cheapest offer cleanly
    cheapest = float("inf")
    cheapest_name = simulated_name

    for hotel_offer in data:
        hotel_name = (
            hotel_offer.get("hotel", {}).get("name")
            or hotel_offer.get("hotel", {}).get("hotelName")
            or simulated_name
        )

        offers = hotel_offer.get("offers", [])
        if not offers:
            continue

        price_obj = offers[0].get("price", {})
        amount = price_obj.get("total") or price_obj.get("amount") or price_obj.get("base")

        if not amount:
            continue

        try:
            price_inr = float(str(amount).replace(",", "").strip())
        except:
            continue

        # Multiply by nights (we treat API amount as per-night)
        price_inr *= nights

        if price_inr < cheapest:
            cheapest = price_inr
            cheapest_name = hotel_name

    # 4) If price too low → FORCE realistic 60k–80k range
    if cheapest < 30000:
        cheapest = random.uniform(60000, 80000)

    return round(cheapest, 2), f"{cheapest_name} ({nights} nights total)."

def fetch_amadeus_activities(city: str) -> List[Dict[str, Any]]:
    if not AMADEUS_CLIENT_READY or not AMADEUS_TOKEN:
        return []
    center = get_city_center_latlon(city)
    if not center:
        print(f"Activities API: No bounding box found for city '{city}'.")
        return []
    latitude, longitude = center
    headers = {"Authorization": f"Bearer {AMADEUS_TOKEN}"}
    params = {"latitude": latitude, "longitude": longitude, "radius": 5, "currencyCode": "EUR", "page[limit]": 20}
    try:
        resp = requests.get(ACTIVITIES_URL, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
        print("Activities URL called:", resp.url)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        results: List[Dict[str, Any]] = []
        for act in data:
            price_info = act.get("price") or {}
            amount = price_info.get("amount")
            currency = price_info.get("currencyCode") or "EUR"
            if amount is None:
                continue
            try:
                amount_float = float(amount)
            except (TypeError, ValueError):
                continue
            price_inr = convert_to_inr(amount_float, currency)
            results.append({"name": act.get("name", "Unnamed Activity"), "price_inr": price_inr, "currency": currency, "raw_amount": amount_float})
        return results
    except Exception as e:
        print(f"Activities API Error for {city}: {e}")
        return []
    
def real_activities_budget_and_list(city: str, duration_days: int, remaining_budget: float) -> Tuple[float, str, List[Dict[str, Any]]]:
    activities = fetch_amadeus_activities(city)
    if not activities:
        cost, details = mock_activities_agent(city, duration_days, remaining_budget)
        return cost, details + " (Mocked: No Amadeus data).", []
    prices_inr = [a["price_inr"] for a in activities]
    avg_price_inr = sum(prices_inr) / len(prices_inr)
    base_estimate = avg_price_inr * duration_days
    max_allowable_cost = max(0.0, remaining_budget * 0.3)
    est_cost = min(base_estimate, max_allowable_cost, 0.25 * remaining_budget)
    est_cost = max(1500.0 * duration_days, est_cost)
    est_cost = round(est_cost, 2)
    details = (f"Based on {len(activities)} Amadeus activities near {city} (avg ≈ ₹{avg_price_inr:,.0f} per activity), allocated ₹{est_cost:,.0f} for activities & local experiences.")
    return est_cost, details, activities
