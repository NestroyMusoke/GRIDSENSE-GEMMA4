import requests
import json
import os
import re
import time
import random
from datetime import datetime
from typing import Optional, Dict, List
from dotenv import load_dotenv

from dotenv import load_dotenv
from pathlib import Path

# Ensure .env is always loaded from project root
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)

# ══════════════════════════════════════════════════════════════════════════════
# API KEY POOLS — add all your keys here (blanks are ignored)
# ══════════════════════════════════════════════════════════════════════════════

GEMINI_KEYS = [k for k in [
    os.getenv("GEMINI_API_KEY_1", os.getenv("GEMINI_API_KEY", "")),
    os.getenv("GEMINI_API_KEY_2", ""),
    os.getenv("GEMINI_API_KEY_3", ""),
] if k]

OPENROUTER_KEYS = [k for k in [
    os.getenv("OPENROUTER_KEY_1", os.getenv("OPENROUTER_API_KEY", "")),
    os.getenv("OPENROUTER_KEY_2", ""),
    os.getenv("OPENROUTER_KEY_3", ""),
    os.getenv("OPENROUTER_KEY_4", ""),
    os.getenv("OPENROUTER_KEY_5", ""),
    os.getenv("OPENROUTER_KEY_6", ""),
] if k]

NVIDIA_KEYS = [k for k in [
    os.getenv("NVIDIA_KEY_1", os.getenv("NVIDIA_API_KEY", "")),
    os.getenv("NVIDIA_KEY_2", ""),
    os.getenv("NVIDIA_KEY_3", ""),
    os.getenv("NVIDIA_KEY_4", ""),
] if k]

GROQ_KEYS = [k for k in [
    os.getenv("GROQ_KEY_1", os.getenv("GROQ_API_KEY", "")),
    os.getenv("GROQ_KEY_2", ""),
    os.getenv("GROQ_KEY_3", ""),
] if k]

# ── Model config ──────────────────────────────────────────────────────────────
GEMINI_MODEL       = "gemma-4-26b-a4b-it"
GEMINI_URL         = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
NVIDIA_URL         = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_MODEL       = "google/gemma-4-31b-it"
GROQ_URL           = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL         = "gemma2-9b-it"

OPENROUTER_MODELS  = [
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-3-27b-it:free",
]

# ── Colab/Kaggle LoRA server (set GRIDSENSE_COLAB_URL in .env for demo mode) ──
COLAB_URL = os.getenv("GRIDSENSE_COLAB_URL", "").rstrip("/")

# ── Rate limit tracker ────────────────────────────────────────────────────────
_rate_limited_until: Dict[str, float] = {}
_failed_keys: set = set()

def _key_ok(key: str) -> bool:
    if key in _failed_keys:
        return False
    unlock = _rate_limited_until.get(key, 0)
    if time.time() < unlock:
        return False
    return True

def _mark_limited(key: str, cooldown: int = 3600):
    _rate_limited_until[key] = time.time() + cooldown
    print(f"[GridSense] Key ...{key[-6:]} rate-limited for {cooldown}s")

def _pick(pool: list) -> Optional[str]:
    available = [k for k in pool if _key_ok(k)]
    return random.choice(available) if available else None

# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are GridSense, a power outage prediction AI for neighborhoods worldwide.

Return ONLY valid JSON. No markdown. No backticks. No extra text before or after.

Probability calibration:
- 0-25: weak or single ambiguous signal, stable weather
- 26-45: one clear signal, stable weather
- 46-65: multiple signals or strong single signal with weather support
- 66-80: multiple corroborating signals plus weather
- 81-90: strong visual evidence plus multiple signals plus severe weather
- 91+: only when transformer failure directly observed by multiple independent sources

Required JSON keys (no extras, no missing):
{
  "probability": integer 0-100,
  "confidence": "INSUFFICIENT" or "LOW" or "MEDIUM" or "HIGH",
  "explanation": "one specific sentence citing actual signals from the report",
  "countdown": "power likely out in X to Y minutes" or null,
  "actions": ["5 personalized action strings based on user devices and priorities"],
  "reasoning_trace": ["6 numbered reasoning steps"],
  "regional_grid_context": "one sentence about local grid infrastructure",
  "weekly_heatmap": {"Monday": [24 integers 0-100], "Tuesday": [24 integers], "Wednesday": [24 integers], "Thursday": [24 integers], "Friday": [24 integers], "Saturday": [24 integers], "Sunday": [24 integers]},
  "regional_grid": [{"name": "neighborhood name", "is_user_area": true or false, "probability": integer 0-100}],
  "signal_strength": "WEAK" or "MODERATE" or "STRONG" or "CRITICAL",
  "memory_influence": "NONE" or "LOW" or "MEDIUM" or "HIGH",
  "utility_future_note": "one sentence about how utility data would improve this"
}

Rules:
- actions must reference the user's specific devices and priorities if provided
- reasoning_trace must show 6 numbered steps
- regional_grid must have 12 to 18 neighborhood objects for the user's city
- weekly_heatmap values must reflect realistic outage patterns for that city and time of week
- Be honest. Never fabricate signals. State uncertainty clearly."""


# ══════════════════════════════════════════════════════════════════════════════
# BRAIN IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

from brain.memory import (
    get_similar_past_reports, save_report,
    format_memory_for_prompt, get_neighborhood_accuracy
)
from brain.weather import get_weather
from brain.validator import validate_input
from brain.translator import detect_and_translate


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYZE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def analyze(
    image_path: Optional[str] = None,
    video_result: Optional[Dict] = None,
    text_report: Optional[str] = None,
    user_profile: Optional[Dict] = None,
    city: str = "Unknown",
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    neighborhood: Optional[str] = None
) -> Dict:

    if user_profile is None:
        user_profile = {}

    original_text = text_report or ""
    detected_lang = "en"

    if text_report:
        try:
            translated_text, detected_lang, lang_name = detect_and_translate(text_report)
            if detected_lang != "en":
                text_report = translated_text
        except Exception:
            pass

    validation = validate_input(text_report or "")

    if not validation["valid"]:
        return {
            "probability": 0,
            "confidence": "INSUFFICIENT",
            "explanation": validation["guidance"],
            "countdown": None,
            "actions": [],
            "reasoning_trace": ["Input rejected by validation layer."],
            "validation_rejected": True,
            "rejection_reason": validation["rejection_reason"],
            "guidance_message": validation["guidance"]
        }

    # ── Weather ───────────────────────────────────────────────────────────────
    weather = {}
    if lat and lon:
        try:
            weather = get_weather(lat, lon)
        except Exception:
            pass

    # ── Memory / RAG ──────────────────────────────────────────────────────────
    past_reports          = []
    memory_context        = ""
    neighborhood_accuracy = {}

    if lat and lon:
        try:
            past_reports          = get_similar_past_reports(lat, lon, radius_km=1.5, limit=5)
            memory_context        = format_memory_for_prompt(past_reports)
            neighborhood_accuracy = get_neighborhood_accuracy(lat, lon)
        except Exception:
            pass

    user_message = _build_message(
        city=city, neighborhood=neighborhood or "Unknown area",
        text_report=text_report, original_text=original_text,
        detected_lang=detected_lang, weather=weather,
        user_profile=user_profile, memory_context=memory_context,
        video_result=video_result, validation=validation
    )

    # ── Inference priority chain ───────────────────────────────────────────────
    # 1. Fine-tuned LoRA on Kaggle/Colab (demo mode — best when running)
    result = _try_colab_lora(user_message)

    # 2. Google AI Studio Gemini
    if result is None:
        result = _try_gemini(user_message, image_path)

    # 3. OpenRouter models
    if result is None:
        result = _try_openrouter(user_message, image_path, video_result)

    # 4. NVIDIA NIM
    if result is None:
        result = _try_nvidia(user_message, image_path)

    # 5. Groq Gemma 2
    if result is None:
        result = _try_groq(user_message)

    # 6. Static fallback
    if result is None:
        result = _fallback_response("All inference providers temporarily unavailable")

    # ── Save to memory ────────────────────────────────────────────────────────
    report_id = None
    if lat and lon and text_report and validation["valid"]:
        try:
            report_id = save_report(
                lat=lat, lon=lon, city=city,
                neighborhood=neighborhood or "Unknown",
                report_text=original_text,
                weather_condition=weather.get("current_condition", "Unknown"),
                predicted_probability=result.get("probability", 30),
                time_of_day=datetime.utcnow().strftime("%H:%M"),
                day_of_week=datetime.utcnow().strftime("%A"),
                signal_keywords=_extract_keywords(text_report),
                confidence_level=result.get("confidence", "LOW")
            )
        except Exception:
            pass

    result["report_id"]                  = report_id
    result["detected_language"]          = detected_lang
    result["memory_reports_used"]        = len(past_reports)
    result["weather_risk"]               = weather.get("grid_risk_label", "UNKNOWN")
    result["weather_data"]               = weather
    result["validation_signal_type"]     = validation["signal_type"]
    result["validation_signal_strength"] = validation["keyword_matches"]
    result["neighborhood_accuracy"]      = neighborhood_accuracy

    if validation.get("guidance"):
        result["guidance_message"] = validation["guidance"]

    return result


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE PROVIDERS
# ══════════════════════════════════════════════════════════════════════════════

def _try_colab_lora(user_message: str) -> Optional[Dict]:
    """
    Calls the fine-tuned Gemma 4 E2B LoRA model running on Kaggle/Colab
    via ngrok tunnel. Only active when GRIDSENSE_COLAB_URL is set in .env.
    Falls through silently if server is unreachable so the API key chain
    takes over automatically.
    """
    if not COLAB_URL:
        return None

    try:
        print(f"[GridSense] Trying LoRA server ({COLAB_URL})...")

        # Health check first — fast 5s timeout so we fail quickly if down
        try:
            health = requests.get(f"{COLAB_URL}/health", timeout=5)
            if health.status_code != 200:
                print("[GridSense] LoRA server unhealthy — skipping")
                return None
        except Exception:
            print("[GridSense] LoRA server unreachable — falling back to APIs")
            return None

        # Main inference request
        resp = requests.post(
            f"{COLAB_URL}/analyze",
            headers={"Content-Type": "application/json"},
            json={"user_message": user_message},
            timeout=400
        )

        if resp.status_code == 200:
            result = resp.json()

            # Server returned an error dict without a valid prediction
            if "error" in result and "probability" not in result:
                print(f"[GridSense] LoRA server error: {result.get('error')}")
                # Try to salvage from raw_text if the model output was valid
                # JSON but the server's strict parser rejected it
                if "raw_text" in result:
                    salvaged = _parse_json(result["raw_text"])
                    if salvaged:
                        salvaged["brain_source"] = "kaggle_lora_gemma4_e2b_salvaged"
                        print("[GridSense] ✅ LoRA (salvaged from raw_text)")
                        return salvaged
                return None

            # Validate required keys are present
            required = {"probability", "confidence", "explanation", "actions",
                        "reasoning_trace", "regional_grid", "weekly_heatmap"}
            if not required.issubset(result.keys()):
                print("[GridSense] LoRA result missing required keys — skipping")
                return None

            print(f"[GridSense] ✅ Kaggle LoRA Gemma 4 E2B — "
                  f"prob={result.get('probability')} conf={result.get('confidence')}")
            return result

        elif resp.status_code == 422:
            # Server got a response but JSON parsing failed — try raw_text
            data = resp.json()
            if "raw_text" in data:
                salvaged = _parse_json(data["raw_text"])
                if salvaged:
                    salvaged["brain_source"] = "kaggle_lora_gemma4_e2b_salvaged"
                    print("[GridSense] ✅ LoRA (salvaged from 422 raw_text)")
                    return salvaged
            return None

        else:
            print(f"[GridSense] LoRA server returned {resp.status_code} — skipping")
            return None

    except requests.exceptions.Timeout:
        print("[GridSense] LoRA server timed out — falling back to APIs")
        return None
    except Exception as e:
        print(f"[GridSense] LoRA server error: {e} — falling back to APIs")
        return None


def _try_gemini(user_message: str, image_path=None) -> Optional[Dict]:
    key = _pick(GEMINI_KEYS)
    if not key:
        return None

    try:
        print(f"[GridSense] Trying Google AI Studio ({GEMINI_MODEL})...")
        parts = [{"text": f"{SYSTEM_PROMPT}\n\n{user_message}"}]

        if image_path:
            try:
                import base64
                with open(image_path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode()
                ext = os.path.splitext(image_path)[1].lower().replace(".", "")
                if ext == "jpg": ext = "jpeg"
                parts.append({"inline_data": {"mime_type": f"image/{ext}", "data": img_b64}})
            except Exception as e:
                print(f"[GridSense] Image attach failed: {e}")

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2000}
        }

        resp = requests.post(
            f"{GEMINI_URL}?key={key}",
            headers={"Content-Type": "application/json"},
            json=payload, timeout=90
        )

        if resp.status_code == 429:
            _mark_limited(key)
            return None
        if resp.status_code != 200:
            print(f"[GridSense] Gemini error {resp.status_code}")
            return None

        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        result = _parse_json(text)
        if result:
            result["brain_source"] = "google_aistudio_gemma4"
            print("[GridSense] ✅ Google AI Studio")
        return result

    except Exception as e:
        print(f"[GridSense] Gemini exception: {e}")
        return None


def _try_openrouter(user_message: str, image_path=None, video_result=None) -> Optional[Dict]:
    """Try each OpenRouter key × each model until one succeeds."""
    if not OPENROUTER_KEYS:
        return None

    messages = _build_openrouter_messages(user_message, image_path, video_result)

    for model in OPENROUTER_MODELS:
        for attempt in range(len(OPENROUTER_KEYS)):
            key = _pick(OPENROUTER_KEYS)
            if not key:
                print("[GridSense] All OpenRouter keys exhausted")
                return None

            try:
                print(f"[GridSense] OpenRouter {model} ...")
                payload = {
                    "model": model,
                    "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                    "temperature": 0.1,
                    "max_tokens": 2000,
                    "stream": True,
                }
                resp = requests.post(
                    OPENROUTER_URL,
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://gridsense.app",
                        "X-Title": "GridSense"
                    },
                    json=payload, timeout=90, stream=True
                )

                if resp.status_code == 429:
                    _mark_limited(key)
                    continue
                if resp.status_code != 200:
                    print(f"[GridSense] OR {model} → {resp.status_code}")
                    continue

                full = ""
                for line in resp.iter_lines():
                    if line:
                        decoded = line.decode("utf-8")
                        if decoded.startswith("data: ") and decoded != "data: [DONE]":
                            try:
                                token = json.loads(decoded[6:])["choices"][0]["delta"].get("content", "")
                                full += token
                            except Exception:
                                pass

                result = _parse_json(full)
                if result:
                    result["brain_source"] = f"openrouter_{model}"
                    print(f"[GridSense] ✅ OpenRouter {model}")
                    return result

            except requests.exceptions.Timeout:
                print(f"[GridSense] OR {model} timeout")
                continue
            except Exception as e:
                print(f"[GridSense] OR {model} error: {e}")
                continue

    return None


def _try_nvidia(user_message: str, image_path=None) -> Optional[Dict]:
    key = _pick(NVIDIA_KEYS)
    if not key:
        return None

    try:
        print(f"[GridSense] Trying NVIDIA NIM ({NVIDIA_MODEL})...")
        messages: list = [{"role": "user", "content": user_message}]

        if image_path:
            try:
                import base64
                with open(image_path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode()
                ext = os.path.splitext(image_path)[1].lower().replace(".", "")
                if ext == "jpg": ext = "jpeg"
                messages = [{"role": "user", "content": [
                    {"type": "text", "text": user_message},
                    {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{img_b64}"}}
                ]}]
            except Exception:
                pass

        payload = {
            "model": NVIDIA_MODEL,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            "temperature": 0.1, "max_tokens": 2000, "stream": True
        }

        resp = requests.post(
            NVIDIA_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload, timeout=120, stream=True
        )

        if resp.status_code in [429, 502, 503]:
            _mark_limited(key, 1800)
            return None
        if resp.status_code != 200:
            return None

        full = ""
        for line in resp.iter_lines():
            if line:
                decoded = line.decode("utf-8")
                if decoded.startswith("data: ") and decoded != "data: [DONE]":
                    try:
                        token = json.loads(decoded[6:])["choices"][0]["delta"].get("content", "")
                        full += token
                    except Exception:
                        pass

        result = _parse_json(full)
        if result:
            result["brain_source"] = "nvidia_nim_gemma4"
            print("[GridSense] ✅ NVIDIA NIM")
        return result

    except Exception as e:
        print(f"[GridSense] NVIDIA error: {e}")
        return None


def _try_groq(user_message: str) -> Optional[Dict]:
    """Groq fallback — Gemma 2 9B is fast and reliable."""
    key = _pick(GROQ_KEYS)
    if not key:
        return None

    try:
        print("[GridSense] Trying Groq (Gemma 2 9B)...")
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.1,
            "max_tokens": 2000,
        }
        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload, timeout=60
        )
        if resp.status_code == 429:
            _mark_limited(key, 60)
            return None
        if resp.status_code != 200:
            return None

        content = resp.json()["choices"][0]["message"]["content"]
        result = _parse_json(content)
        if result:
            result["brain_source"] = "groq_gemma2"
            print("[GridSense] ✅ Groq Gemma 2")
        return result

    except Exception as e:
        print(f"[GridSense] Groq error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _parse_json(text: str) -> Optional[Dict]:
    try:
        text = text.replace("```json", "").replace("```", "").strip()
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            text = match.group(0)
        result = json.loads(text)
        if "probability" in result:
            prob = result["probability"]
            if isinstance(prob, (float, int)):
                result["probability"] = max(0, min(100, int(round(float(prob)))))
        required = {"probability", "confidence", "explanation", "actions",
                    "reasoning_trace", "regional_grid", "weekly_heatmap"}
        if not required.issubset(result.keys()):
            return None
        return result
    except Exception:
        return None


def _build_openrouter_messages(user_message, image_path, video_result):
    if image_path:
        try:
            import base64
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            ext = os.path.splitext(image_path)[1].lower().replace(".", "")
            if ext == "jpg": ext = "jpeg"
            return [{"role": "user", "content": [
                {"type": "text", "text": user_message},
                {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{img_b64}"}}
            ]}]
        except Exception:
            pass

    elif video_result and video_result.get("selected_frames"):
        try:
            import base64
            parts = [{"type": "text", "text": user_message}]
            for fp in video_result["selected_frames"][:3]:
                with open(fp, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode()
                parts.append({"type": "image_url",
                               "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})
            return [{"role": "user", "content": parts}]
        except Exception:
            pass

    return [{"role": "user", "content": user_message}]


def _build_message(city, neighborhood, text_report, original_text,
                   detected_lang, weather, user_profile,
                   memory_context, video_result, validation) -> str:
    parts = [
        f"Location: {city}, {neighborhood}",
        f"Time: {datetime.utcnow().strftime('%A %H:%M UTC')}",
    ]

    if detected_lang != "en" and original_text:
        parts += [f"Report (original {detected_lang}): {original_text}",
                  f"Report (translated): {text_report}"]
    elif text_report:
        parts.append(f"Report: {text_report}")

    parts.append(f"Signal validation: type={validation['signal_type']}, "
                 f"keywords={validation['keyword_matches']}, "
                 f"tone={validation.get('confidence_tone', 'unknown')}")

    if video_result:
        parts.append(f"Video: {video_result.get('frame_summary', '')}")
        if video_result.get("flicker_score", 0) > 0.1:
            parts.append(f"Flicker score: {video_result['flicker_score']} (0=stable 1=severe)")
        if video_result.get("audio_transcript"):
            parts.append(f"Audio transcript: {video_result['audio_transcript']}")

    if weather.get("weather_summary"):
        parts.append(f"Current weather: {weather['weather_summary']}")
        parts.append(f"Weather grid risk: {weather.get('grid_risk_label', 'UNKNOWN')} "
                     f"(score {weather.get('grid_risk_contribution', 0)}/100)")

    if memory_context and "No similar" not in memory_context:
        parts.append(memory_context[:500])

    if user_profile:
        parts += [f"User priorities: {user_profile.get('priorities', [])}",
                  f"User devices: {user_profile.get('devices', [])}"]

    parts.append("\nAnalyze all signals. Return ONLY valid JSON.")
    return "\n".join(parts)


def _extract_keywords(text: str) -> list:
    keywords = []
    t = text.lower()
    for w in ["flicker", "hum", "transformer", "crew", "van", "smell", "burning",
              "sparks", "surge", "dim", "generator", "inverter", "brownout",
              "schedule", "maintenance", "load shed", "tripped", "pole", "wire"]:
        if w in t:
            keywords.append(w)
    return keywords


def _fallback_response(error: str) -> Dict:
    return {
        "probability": 35,
        "confidence": "LOW",
        "explanation": "Analysis temporarily unavailable. Monitoring neighborhood patterns based on typical patterns for this time of day.",
        "countdown": None,
        "actions": [
            "Keep your phone and essential devices charged now",
            "Note any unusual sounds from nearby transformers",
            "Check your local utility outage map if available",
            "Tell neighbors to also report what they observe",
            "GridSense will retry analysis on your next report"
        ],
        "reasoning_trace": [
            "1. Visual signals: None detected — no image provided",
            "2. Text signals: Analysis service temporarily unavailable",
            "3. Weather: Unknown — could not reach weather service",
            "4. Memory: No past reports retrieved",
            "5. Geographic fencing: Location recorded for future analysis",
            f"6. Final synthesis: Fallback — {error}"
        ],
        "regional_grid_context": "Monitoring local grid conditions as data becomes available.",
        "weekly_heatmap": {
            day: [20 if (h < 6 or h > 22) else 30 for h in range(24)]
            for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        },
        "regional_grid": [
            {"name": "Your area", "is_user_area": True, "probability": 35}
        ],
        "signal_strength": "WEAK",
        "memory_influence": "NONE",
        "utility_future_note": "Real-time utility feed from your provider would unlock 40% more accurate predictions.",
        "brain_source": "fallback",
        "error": error
    }