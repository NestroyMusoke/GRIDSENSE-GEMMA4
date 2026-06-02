from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import json, tempfile, os, time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from brain.analyzer import analyze
from brain.memory import (
    confirm_outcome, get_similar_past_reports,
    get_neighborhood_accuracy, get_recent_reports_for_map
)

app = FastAPI(
    title="GridSense API",
    description="Neighborhood power outage prediction — Gemma 4 multimodal + RAG memory",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'frontend')
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
async def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/health")
async def health():
    from brain.analyzer import GEMINI_KEYS, OPENROUTER_KEYS, NVIDIA_KEYS, GROQ_KEYS
    return {
        "status": "online",
        "version": "2.0.0",
        "providers": {
            "gemini": len(GEMINI_KEYS),
            "openrouter": len(OPENROUTER_KEYS),
            "nvidia": len(NVIDIA_KEYS),
            "groq": len(GROQ_KEYS),
        },
        "capabilities": [
            "multimodal_photo", "multimodal_video", "voice_transcription",
            "weather_fusion", "rag_memory", "multilingual",
            "6_accuracy_layers", "multi_provider_fallback"
        ]
    }


@app.post("/analyze")
async def analyze_report(
    text_report: str   = Form(default=""),
    city: str          = Form(default="Unknown"),
    neighborhood: str  = Form(default=""),
    lat: float         = Form(default=None),
    lon: float         = Form(default=None),
    user_profile: str  = Form(default="{}"),
    image: UploadFile  = File(default=None),
    video: UploadFile  = File(default=None)
):
    t0 = time.time()
    image_path = None
    video_path = None
    video_result = None

    # ── Image handling ────────────────────────────────────────────────────────
    if image and image.filename:
        suffix = Path(image.filename).suffix.lower()
        if suffix not in {'.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif'}:
            raise HTTPException(400, "Unsupported image format")
        content = await image.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            image_path = tmp.name

    # ── Video handling ────────────────────────────────────────────────────────
    if video and video.filename:
        suffix = Path(video.filename).suffix.lower()
        if suffix not in {'.mp4', '.mov', '.webm', '.avi', '.mkv', '.m4v'}:
            raise HTTPException(400, "Unsupported video format")
        content = await video.read()
        if len(content) > 50 * 1024 * 1024:
            raise HTTPException(400, "Video exceeds 50MB limit")
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            video_path = tmp.name

    try:
        # ── Video processing ──────────────────────────────────────────────────
        if video_path:
            try:
                from brain.video_processor import process_video
                video_result = process_video(video_path)
            except Exception as e:
                print(f"[GridSense] Video processing failed: {e}")

        # ── Profile parsing ───────────────────────────────────────────────────
        try:
            profile = json.loads(user_profile)
        except Exception:
            profile = {}

        # ── Core analysis ─────────────────────────────────────────────────────
        result = analyze(
            image_path=image_path,
            video_result=video_result,
            text_report=text_report,
            user_profile=profile,
            city=city,
            lat=lat,
            lon=lon,
            neighborhood=neighborhood or None
        )

        result["processing_time_ms"] = round((time.time() - t0) * 1000)
        result["input_type"] = (
            "video_multimodal" if video_path else
            "photo_multimodal" if image_path else
            "text_only"
        )

        return result

    finally:
        for path in [image_path, video_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception:
                    pass
        if video_result:
            try:
                from brain.video_processor import cleanup_temp_files
                cleanup_temp_files(video_result)
            except Exception:
                pass


@app.post("/confirm-outcome")
async def confirm_prediction_outcome(report_id: int, outcome: str):
    valid = {"outage_occurred", "no_outage", "partial_outage"}
    if outcome not in valid:
        raise HTTPException(400, f"outcome must be one of {valid}")
    confirm_outcome(report_id, outcome)
    return {"status": "confirmed", "report_id": report_id, "outcome": outcome}


@app.get("/map-data")
async def get_map_data(lat: float, lon: float, radius_km: float = 5.0):
    reports  = get_similar_past_reports(lat, lon, radius_km, limit=50)
    accuracy = get_neighborhood_accuracy(lat, lon)
    points   = []
    for r in reports:
        risk = "high" if r["predicted_probability"] >= 65 else \
               "medium" if r["predicted_probability"] >= 40 else "low"
        points.append({
            "lat": r.get("lat", lat),
            "lon": r.get("lon", lon),
            "probability": r["predicted_probability"],
            "risk_level": risk,
            "timestamp": r["timestamp"],
            "distance_km": r["distance_km"],
            "confirmed": r["outcome_confirmed"]
        })
    return {
        "center": {"lat": lat, "lon": lon},
        "radius_km": radius_km,
        "data_points": points,
        "neighborhood_accuracy": accuracy,
        "total_reports": len(reports)
    }


@app.get("/neighborhood-stats")
async def neighborhood_stats(lat: float, lon: float):
    accuracy = get_neighborhood_accuracy(lat, lon)
    recent   = get_similar_past_reports(lat, lon, radius_km=1.5, limit=10)
    n = len(recent)
    return {
        "accuracy": accuracy,
        "recent_reports": n,
        "last_report_time": recent[0]["timestamp"] if recent else None,
        "learning_status": (
            "LEARNING"     if n < 5 else
            "CALIBRATING"  if n < 15 else
            "TRAINED"
        )
    }

import os

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 7860))
    )
