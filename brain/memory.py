import sqlite3
import json
import os
import math
from datetime import datetime
from typing import List, Dict, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'gridsense_memory.db')

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS neighborhood_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            city TEXT,
            neighborhood TEXT,
            report_text TEXT,
            weather_condition TEXT,
            predicted_probability INTEGER,
            actual_outcome TEXT,
            outcome_confirmed INTEGER DEFAULT 0,
            time_of_day TEXT,
            day_of_week TEXT,
            signal_keywords TEXT,
            confidence_level TEXT
        )
    """)
    conn.commit()
    conn.close()

def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = math.sin(dlat/2)**2 + math.cos(lat1_r)*math.cos(lat2_r)*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def save_report(lat: float, lon: float, city: str, neighborhood: str,
                report_text: str, weather_condition: str,
                predicted_probability: int, time_of_day: str,
                day_of_week: str, signal_keywords: list,
                confidence_level: str) -> int:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO neighborhood_reports
        (timestamp, lat, lon, city, neighborhood, report_text,
         weather_condition, predicted_probability, time_of_day,
         day_of_week, signal_keywords, confidence_level)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(), lat, lon, city, neighborhood,
        report_text, weather_condition, predicted_probability,
        time_of_day, day_of_week, json.dumps(signal_keywords), confidence_level
    ))
    report_id = c.lastrowid
    conn.commit()
    conn.close()
    return report_id

def confirm_outcome(report_id: int, actual_outcome: str):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        UPDATE neighborhood_reports
        SET actual_outcome = ?, outcome_confirmed = 1
        WHERE id = ?
    """, (actual_outcome, report_id))
    conn.commit()
    conn.close()

def get_similar_past_reports(lat: float, lon: float,
                              radius_km: float = 1.5,
                              limit: int = 8) -> List[Dict]:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    lat_range = radius_km / 111.0
    lon_range = radius_km / (111.0 * abs(math.cos(math.radians(lat))) + 0.001)
    c.execute("""
        SELECT id, timestamp, lat, lon, neighborhood, report_text,
               weather_condition, predicted_probability, actual_outcome,
               outcome_confirmed, time_of_day, day_of_week, confidence_level
        FROM neighborhood_reports
        WHERE lat BETWEEN ? AND ?
        AND lon BETWEEN ? AND ?
        ORDER BY timestamp DESC
        LIMIT 50
    """, (lat - lat_range, lat + lat_range,
          lon - lon_range, lon + lon_range))
    rows = c.fetchall()
    conn.close()
    nearby = []
    for row in rows:
        dist = haversine_distance(lat, lon, row[2], row[3])
        if dist <= radius_km:
            nearby.append({
                "id": row[0],
                "timestamp": row[1],
                "lat": row[2],
                "lon": row[3],
                "distance_km": round(dist, 2),
                "neighborhood": row[4],
                "report": row[5],
                "weather": row[6],
                "predicted_probability": row[7],
                "actual_outcome": row[8],
                "outcome_confirmed": bool(row[9]),
                "time_of_day": row[10],
                "day_of_week": row[11],
                "confidence": row[12]
            })
    nearby.sort(key=lambda x: x["distance_km"])
    return nearby[:limit]

def get_neighborhood_accuracy(lat: float, lon: float,
                               radius_km: float = 1.5) -> Dict:
    reports = get_similar_past_reports(lat, lon, radius_km, limit=100)
    confirmed = [r for r in reports if r["outcome_confirmed"]]
    if len(confirmed) < 3:
        return {
            "accuracy_available": False,
            "total_confirmed": len(confirmed),
            "message": f"Only {len(confirmed)} confirmed outcomes. Need 3+ for accuracy metrics."
        }
    correct = sum(1 for r in confirmed
                  if (r["predicted_probability"] >= 65 and r["actual_outcome"] == "outage_occurred") or
                     (r["predicted_probability"] < 65 and r["actual_outcome"] == "no_outage"))
    accuracy = correct / len(confirmed)
    outage_reports = [r for r in confirmed if r["actual_outcome"] == "outage_occurred"]
    return {
        "accuracy_available": True,
        "accuracy_percent": round(accuracy * 100, 1),
        "total_confirmed": len(confirmed),
        "total_outages_confirmed": len(outage_reports),
        "message": f"Based on {len(confirmed)} confirmed predictions in this neighborhood"
    }

def format_memory_for_prompt(past_reports: List[Dict]) -> str:
    if not past_reports:
        return "No similar past reports found within 1.5km of this location."
    lines = [f"RETRIEVED MEMORY: {len(past_reports)} similar past reports from within 1.5km:\n"]
    for i, r in enumerate(past_reports, 1):
        outcome_str = ""
        if r["outcome_confirmed"]:
            outcome_str = f" ACTUAL OUTCOME: {r['actual_outcome']}"
        time_ago = _time_ago(r["timestamp"])
        lines.append(
            f"Past Event {i} ({time_ago}, {r['distance_km']}km away):\n"
            f"  Report: {r['report']}\n"
            f"  Weather: {r['weather']}\n"
            f"  Prediction: {r['predicted_probability']}%{outcome_str}\n"
        )
    lines.append("\nUSE THESE PAST EVENTS to calibrate your current prediction.")
    return "\n".join(lines)

def _time_ago(timestamp: str) -> str:
    try:
        dt = datetime.fromisoformat(timestamp)
        diff = datetime.utcnow() - dt
        if diff.days > 0:
            return f"{diff.days} days ago"
        hours = diff.seconds // 3600
        if hours > 0:
            return f"{hours} hours ago"
        minutes = diff.seconds // 60
        return f"{minutes} minutes ago"
    except:
        return "recently"

def get_recent_reports_for_map(lat: float, lon: float,
                                radius_km: float = 5.0,
                                limit: int = 50) -> List[Dict]:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT id, timestamp, lat, lon, neighborhood,
               predicted_probability, outcome_confirmed, actual_outcome
        FROM neighborhood_reports
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))
    rows = c.fetchall()
    conn.close()
    return [{
        "id": row[0],
        "timestamp": row[1],
        "lat": row[2],
        "lon": row[3],
        "neighborhood": row[4],
        "predicted_probability": row[5],
        "outcome_confirmed": bool(row[6]),
        "actual_outcome": row[7]
    } for row in rows]