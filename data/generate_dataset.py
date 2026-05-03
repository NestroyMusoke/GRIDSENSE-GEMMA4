import ollama
import json
import random
import time
import os
from datetime import datetime, timedelta

# Always save in the same folder as this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "synthetic_outage_reports.jsonl")

CITIES = {
    "Mukono, Uganda": {
        "lat": 0.3531, "lon": 32.7553,
        "utility": "UMEME",
        "context": "East African grid, frequent transformer overload at peak hours 6-9pm, rainy season causes line trips",
        "neighborhoods": ["Bugolobi", "Ntinda", "Nakawa", "Kireka", "Bweyogerere", "Namanve"]
    },
    "Lagos, Nigeria": {
        "lat": 6.5244, "lon": 3.3792,
        "utility": "EKEDC",
        "context": "Nigerian grid with daily load shedding, transformer theft common, evening peaks 7-10pm most critical",
        "neighborhoods": ["Surulere", "Yaba", "Mushin", "Ikeja", "Lekki", "Victoria Island", "Apapa"]
    },
    "Karachi, Pakistan": {
        "lat": 24.8607, "lon": 67.0011,
        "utility": "K-Electric",
        "context": "K-Electric load shedding zone system, scheduled cuts 8-12 hours in residential areas, summer peak demand critical",
        "neighborhoods": ["Defence", "Clifton", "Gulshan", "Korangi", "Saddar", "Orangi", "Malir"]
    },
    "Johannesburg, South Africa": {
        "lat": -26.2041, "lon": 28.0473,
        "utility": "Eskom",
        "context": "Eskom load shedding Stage 1-6, published schedules sometimes not followed, winter demand peaks June-August",
        "neighborhoods": ["Soweto", "Sandton", "Alexandra", "Roodepoort", "Eldorado Park", "Lenasia"]
    },
    "Manila, Philippines": {
        "lat": 14.5995, "lon": 120.9842,
        "utility": "Meralco",
        "context": "Typhoon season June-November causes distribution damage, brownouts during peak demand, feeder tripping common",
        "neighborhoods": ["Tondo", "Binondo", "Paco", "Pandacan", "Santa Ana", "Makati", "Pasig"]
    },
    "Beirut, Lebanon": {
        "lat": 33.8938, "lon": 35.5018,
        "utility": "EDL + private generators",
        "context": "State electricity 2-4 hours daily, private moteur schedules fill gaps, diesel shortages cause cascading failures",
        "neighborhoods": ["Hamra", "Achrafieh", "Bourj Hammoud", "Nabaa", "Dekwaneh", "Sin el Fil"]
    },
    "Chennai, India": {
        "lat": 13.0827, "lon": 80.2707,
        "utility": "TNEB",
        "context": "TNEB planned maintenance Tuesdays and Saturdays, summer May-June worst period, monsoon line damage September-November",
        "neighborhoods": ["T. Nagar", "Adyar", "Velachery", "Tambaram", "Perambur", "Anna Nagar"]
    },
    "San Francisco, USA": {
        "lat": 37.7749, "lon": -122.4194,
        "utility": "PG&E",
        "context": "PG&E PSPS fire season August-November, high wind shutoffs most common trigger, grid aging in East Bay feeders",
        "neighborhoods": ["Mission", "Tenderloin", "Sunset", "Richmond", "Bayview", "Castro"]
    }
}

SIGNAL_TYPES = [
    "lights flickering in the last few minutes",
    "transformer humming loudly near the junction",
    "utility crew van parked down the street",
    "neighbor said the line crew was working this morning",
    "burning smell near the electrical pole",
    "lights dimming and brightening repeatedly",
    "my inverter kicked in briefly and then restored",
    "power went off and came back three times in the last hour",
    "dark clouds coming from the north, storm approaching",
    "scheduled maintenance notice from the utility on WhatsApp group",
    "saw sparks from the transformer across the road",
    "multiple neighbors reporting flickering on our street",
    "just heard a loud bang from the direction of the substation",
    "power stable but neighbor two blocks away has no power",
    "load shedding started one street over"
]

WEATHER_CONDITIONS = [
    {"condition": "heavy rain approaching", "impact": "high"},
    {"condition": "clear sky, hot day", "impact": "medium"},
    {"condition": "thunderstorm nearby", "impact": "very high"},
    {"condition": "mild overcast", "impact": "low"},
    {"condition": "strong winds 40kmh", "impact": "high"},
    {"condition": "dry harmattan winds", "impact": "medium"},
    {"condition": "temperature 38C peak demand", "impact": "high"}
]

USER_PROFILES = [
    {"priorities": ["work", "laptop"], "devices": ["laptop", "phone", "router"]},
    {"priorities": ["food", "fridge"], "devices": ["refrigerator", "phone"]},
    {"priorities": ["medical", "device"], "devices": ["medical device", "phone", "laptop"]},
    {"priorities": ["child routine", "lighting"], "devices": ["phone", "laptop", "water pump"]},
    {"priorities": ["business", "POS system"], "devices": ["laptop", "phone", "router", "refrigerator"]}
]

GENERATION_PROMPT = """You are generating training data for a power outage prediction system.

Given this neighborhood situation, generate a realistic prediction response.

City: {city}
Utility: {utility}
Grid context: {context}
Neighborhood: {neighborhood}
User report: {report}
Weather: {weather}
User profile priorities: {priorities}
User devices: {devices}
Time of day: {time_of_day}
Day of week: {day_of_week}

Generate a realistic, calibrated prediction. Return ONLY valid JSON:
{{
  "probability": <integer 0-100, be realistic and varied>,
  "confidence": "<INSUFFICIENT|LOW|MEDIUM|HIGH>",
  "explanation": "<one specific sentence referencing the actual signals provided>",
  "countdown": "<'power likely in X to Y minutes' only if probability above 65, else null>",
  "actions": [
    "<action 1 specific to user's devices and priorities>",
    "<action 2 specific to user's devices and priorities>",
    "<action 3 specific to user's devices and priorities>",
    "<action 4 specific to user's devices and priorities>",
    "<action 5 specific to user's devices and priorities>"
  ],
  "reasoning_trace": [
    "Signal 1: <what signal was detected and its weight>",
    "Signal 2: <what signal was detected and its weight>",
    "Signal 3: <weather contribution>",
    "Conclusion: <how signals combined to reach this probability>"
  ],
  "regional_risk": "<one sentence about area-specific grid factors>",
  "signal_strength": "<WEAK|MODERATE|STRONG>"
}}

Rules:
- Vary probabilities realistically: not every report is high risk
- Low signal reports should give 20-40% probability
- Multiple corroborating signals should give 65-85%
- Only give 85%+ when transformer sparks, multiple reports, plus weather
- Actions must reference the specific devices and priorities given
- Make reasoning trace specific to the actual signals provided"""


def generate_example(city_name, city_data):
    num_signals = random.randint(1, 4)
    signals = random.sample(SIGNAL_TYPES, num_signals)
    report = ". ".join(signals) + "."

    weather = random.choice(WEATHER_CONDITIONS)
    profile = random.choice(USER_PROFILES)
    neighborhood = random.choice(city_data["neighborhoods"])

    hour = random.randint(0, 23)
    time_of_day = f"{hour:02d}:00"
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_of_week = random.choice(days)

    prompt = GENERATION_PROMPT.format(
        city=city_name,
        utility=city_data["utility"],
        context=city_data["context"],
        neighborhood=neighborhood,
        report=report,
        weather=weather["condition"],
        priorities=profile["priorities"],
        devices=profile["devices"],
        time_of_day=time_of_day,
        day_of_week=day_of_week
    )

    try:
        response = ollama.generate(
            model="gemma4:e4b",
            prompt=prompt,
            format="json",
            options={"temperature": 0.8, "num_predict": 800}
        )
        prediction = json.loads(response["response"])
        return {
            "input": {
                "city": city_name,
                "utility": city_data["utility"],
                "neighborhood": neighborhood,
                "lat": city_data["lat"] + random.uniform(-0.005, 0.005),
                "lon": city_data["lon"] + random.uniform(-0.005, 0.005),
                "report": report,
                "weather": weather,
                "user_profile": profile,
                "time_of_day": time_of_day,
                "day_of_week": day_of_week,
                "num_signals": num_signals
            },
            "output": prediction
        }
    except Exception as e:
        print(f"  [error] {city_name}: {e}")
        return None


def count_existing():
    """Count how many examples per city are already saved."""
    completed = {}
    if not os.path.exists(OUTPUT_PATH):
        return completed
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                city = ex["input"]["city"]
                completed[city] = completed.get(city, 0) + 1
            except Exception:
                pass
    return completed


def main():
    # 800 total = 100 per city
    # Small enough to finish in one power cycle
    # Large enough to fine-tune on
    target = 800
    per_city = target // len(CITIES)

    print(f"GridSense Dataset Generator")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Target: {target} examples ({per_city} per city)")
    print("-" * 50)

    # Check what is already on disk
    completed = count_existing()
    total_done = sum(completed.values())

    if total_done > 0:
        print(f"RESUMING — {total_done} examples already saved on disk:")
        for city, count in completed.items():
            status = "DONE" if count >= per_city else f"{count}/{per_city}"
            print(f"  {city}: {status}")
    else:
        print("Starting fresh — no existing data found.")

    print("-" * 50)

    # Open in APPEND mode — never overwrites existing data
    with open(OUTPUT_PATH, "a", encoding="utf-8") as outfile:
        for city_name, city_data in CITIES.items():
            already_done = completed.get(city_name, 0)
            remaining = per_city - already_done

            if remaining <= 0:
                print(f"\n[SKIP] {city_name} — already complete ({already_done}/{per_city})")
                continue

            print(f"\n[START] {city_name} — generating {remaining} more examples...")
            city_count = 0

            while city_count < remaining:
                example = generate_example(city_name, city_data)
                if example:
                    # Write AND flush AND fsync every single example
                    # This guarantees it is on physical disk immediately
                    outfile.write(json.dumps(example) + "\n")
                    outfile.flush()
                    os.fsync(outfile.fileno())
                    city_count += 1
                    total_so_far = already_done + city_count
                    if city_count % 5 == 0:
                        print(f"  {total_so_far}/{per_city} saved to disk for {city_name}")
                time.sleep(0.2)

            print(f"[DONE] {city_name} complete.")

    # Final verification
    final_completed = count_existing()
    final_total = sum(final_completed.values())
    print("\n" + "=" * 50)
    print(f"GENERATION COMPLETE")
    print(f"Total examples on disk: {final_total}")
    print(f"File location: {OUTPUT_PATH}")
    for city, count in final_completed.items():
        print(f"  {city}: {count} examples")
    print("=" * 50)


if __name__ == "__main__":
    main()