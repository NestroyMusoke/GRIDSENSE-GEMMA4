import requests
from typing import Dict

def get_weather(lat: float, lon: float) -> Dict:
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": [
                "temperature_2m", "precipitation", "wind_speed_10m",
                "cloud_cover", "weather_code", "wind_gusts_10m",
                "relative_humidity_2m"
            ],
            "hourly": ["precipitation_probability", "wind_speed_10m", "weather_code"],
            "forecast_hours": 6,
            "timezone": "auto"
        }
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        current = data.get("current", {})
        hourly = data.get("hourly", {})
        weather_code = current.get("weather_code", 0)
        condition = _decode_weather_code(weather_code)
        next_6h_precip = hourly.get("precipitation_probability", [0]*6)[:6]
        max_precip_prob = max(next_6h_precip) if next_6h_precip else 0
        next_6h_wind = hourly.get("wind_speed_10m", [0]*6)[:6]
        max_wind = max(next_6h_wind) if next_6h_wind else 0
        grid_risk = _calculate_weather_grid_risk(
            current.get("precipitation", 0),
            current.get("wind_speed_10m", 0),
            current.get("wind_gusts_10m", 0),
            weather_code, max_precip_prob, max_wind
        )
        return {
            "current_condition": condition,
            "temperature_c": current.get("temperature_2m"),
            "precipitation_mm": current.get("precipitation", 0),
            "wind_speed_kmh": current.get("wind_speed_10m", 0),
            "wind_gusts_kmh": current.get("wind_gusts_10m", 0),
            "cloud_cover_percent": current.get("cloud_cover", 0),
            "humidity_percent": current.get("relative_humidity_2m"),
            "max_precipitation_probability_6h": max_precip_prob,
            "max_wind_speed_6h_kmh": max_wind,
            "grid_risk_contribution": grid_risk,
            "grid_risk_label": _risk_label(grid_risk),
            "weather_summary": (
                f"{condition}, {current.get('temperature_2m', '?')}C, "
                f"wind {current.get('wind_speed_10m', 0):.0f} km/h, "
                f"precipitation {current.get('precipitation', 0):.1f}mm. "
                f"Next 6h: max wind {max_wind:.0f} km/h, "
                f"rain probability {max_precip_prob:.0f}%."
            )
        }
    except Exception as e:
        return {
            "current_condition": "Weather data unavailable",
            "grid_risk_contribution": 10,
            "grid_risk_label": "UNKNOWN",
            "weather_summary": "Weather service temporarily unavailable.",
            "error": str(e)
        }

def _decode_weather_code(code: int) -> str:
    if code == 0: return "Clear sky"
    elif code in [1, 2, 3]: return "Partly cloudy"
    elif code in [45, 48]: return "Foggy"
    elif code in [51, 53, 55]: return "Drizzle"
    elif code in [61, 63, 65]: return "Rain"
    elif code in [71, 73, 75]: return "Snow"
    elif code in [80, 81, 82]: return "Rain showers"
    elif code in [95, 96, 99]: return "Thunderstorm"
    else: return "Overcast"

def _calculate_weather_grid_risk(precip, wind, gusts, code, next_precip, next_wind):
    risk = 0
    if precip > 5: risk += 25
    elif precip > 1: risk += 15
    elif precip > 0: risk += 8
    if wind > 50: risk += 30
    elif wind > 30: risk += 20
    elif wind > 15: risk += 10
    if gusts > 70: risk += 20
    elif gusts > 40: risk += 10
    if code in [95, 96, 99]: risk += 30
    elif code in [80, 81, 82]: risk += 15
    if next_precip > 70: risk += 15
    elif next_precip > 40: risk += 8
    if next_wind > 50: risk += 15
    elif next_wind > 30: risk += 8
    return min(100, risk)

def _risk_label(risk: int) -> str:
    if risk >= 60: return "HIGH"
    elif risk >= 35: return "MODERATE"
    elif risk >= 15: return "LOW"
    else: return "MINIMAL"