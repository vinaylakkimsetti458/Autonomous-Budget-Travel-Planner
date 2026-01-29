import os
import random
import requests
from typing import List, Tuple
from utils.helpers import (
    get_city_bbox,
    MEAL_COST_TIERS,
)

GEOAPIFY_PLACES_URL = "https://api.geoapify.com/v2/places"
REQUEST_TIMEOUT = 15

GEOAPIFY_API_KEY = os.getenv("GEOAPIFY_API_KEY")

def real_food_api(city: str, duration_days: int) -> Tuple[float, List[str]]:
    city_bbox = get_city_bbox(city)
    restaurant_names = ["Local Cafe", "Street Food Stall", "Casual Dining", "Regional Specialty", "Fine Food Spot"]
    try:
        if not GEOAPIFY_API_KEY:
            raise Exception("Geoapify API Key not found.")
        if city_bbox:
            bias_param = f'rect:{city_bbox}'
        else:
            raise Exception(f"Bounding box not found for {city}.")
        params = {'categories': 'catering.restaurant,catering.fast_food', 'filter': bias_param, 'limit': 10, 'apiKey': GEOAPIFY_API_KEY}
        response = requests.get(GEOAPIFY_PLACES_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        results = response.json().get('features', [])
        if results:
            restaurant_names = [r['properties'].get('name', 'Unnamed Spot') for r in results if r['properties'].get('name')]
            restaurant_names = list(set(restaurant_names))[:5]
    except Exception as e:
        print(f"Geoapify API failed for {city}: {e}. Using mock data.")
        pass

    total_food_cost = 0.0
    itinerary_names: List[str] = []
    random.shuffle(restaurant_names)
    name_index = 0
    for day in range(1, duration_days + 1):
        for meal_type in ["Breakfast", "Lunch", "Dinner"]:
            if meal_type == "Breakfast":
                tier = "Low"
            elif meal_type == "Lunch":
                tier = random.choice(["Medium", "Medium", "Medium", "Low", "Low"])
            else:
                tier = random.choice(["Medium", "Medium", "Medium", "High", "High"])
            cost = MEAL_COST_TIERS[tier]
            total_food_cost += cost
            name = restaurant_names[name_index % len(restaurant_names)]
            itinerary_names.append(f"Day {day} {meal_type} ({tier} - ₹{cost:,.0f}): {name}")
            name_index += 1
    return round(total_food_cost, 2), itinerary_names