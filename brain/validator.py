import re
from typing import Dict

HIGH_SIGNAL_KEYWORDS = {
    "flicker", "flickering", "hum", "humming", "transformer", "crew", "van",
    "smell", "burning", "sparks", "spark", "surge", "dim", "dimming",
    "brownout", "schedule", "maintenance", "load shed", "load shedding",
    "stage 4", "stage 6", "stage 3", "dark", "gone", "off", "no power",
    "no light", "generator", "kicked in", "inverter", "substation",
    "pole", "line", "wire", "cable", "breaker", "blown", "trip", "tripped"
}

MEDIUM_KEYWORDS = {
    "rumor", "neighbor", "said", "heard", "sometimes", "usually",
    "every evening", "often", "weekly", "always goes"
}

HOUSEHOLD_KEYWORDS = {
    "my socket", "my breaker", "my fuse", "my bulb", "my appliance",
    "my room", "my house only", "only my", "just my", "my wiring",
    "my cord", "only in my", "my fridge stopped", "my phone charger",
    "my outlet", "my switch"
}

PAST_TENSE = {
    "yesterday", "last week", "last month", "earlier today",
    "this morning", "last time", "the other day"
}

PRESENT_TENSE = {"right now", "just", "currently", "happening", "just now"}

TEST_INPUTS = {"test", "testing", "hello", "hi", "asdf", "qwerty", "lol", "idk", ""}

def validate_input(text: str) -> Dict:
    if not text or len(text.strip()) < 4:
        return {
            "valid": False,
            "signal_type": "none",
            "confidence_tone": "unknown",
            "keyword_matches": 0,
            "rejection_reason": "too_short",
            "guidance": "Tell GridSense a little more about what you are noticing."
        }

    text_lower = text.lower().strip()

    if text_lower in TEST_INPUTS:
        return {
            "valid": False,
            "signal_type": "none",
            "confidence_tone": "unknown",
            "keyword_matches": 0,
            "rejection_reason": "test_input",
            "guidance": "Looks like a test. When you are ready, describe what you are actually noticing."
        }

    if re.search(r'(.)\1{5,}', text_lower) or len(re.sub(r'[^a-z]', '', text_lower)) < 3:
        return {
            "valid": False,
            "signal_type": "none",
            "confidence_tone": "unknown",
            "keyword_matches": 0,
            "rejection_reason": "incoherent",
            "guidance": "That did not look like a real report. Try describing what you see or hear."
        }

    has_present = any(p in text_lower for p in PRESENT_TENSE)
    has_past = any(p in text_lower for p in PAST_TENSE)
    if has_past and not has_present:
        return {
            "valid": False,
            "signal_type": "none",
            "confidence_tone": "unknown",
            "keyword_matches": 0,
            "rejection_reason": "not_current",
            "guidance": "GridSense needs real-time signals. Describe what is happening right now."
        }

    household_match = any(kw in text_lower for kw in HOUSEHOLD_KEYWORDS)
    if household_match:
        return {
            "valid": True,
            "signal_type": "household",
            "confidence_tone": "confident",
            "keyword_matches": 0,
            "rejection_reason": None,
            "guidance": "That sounds like it might only be affecting your home. GridSense will watch the neighborhood, but check your own breaker and wiring first."
        }

    high_matches = sum(1 for kw in HIGH_SIGNAL_KEYWORDS if kw in text_lower)
    medium_matches = sum(1 for kw in MEDIUM_KEYWORDS if kw in text_lower)
    total_matches = high_matches + (medium_matches * 0.5)

    hedging = {"maybe", "i think", "possibly", "not sure", "might", "could be", "seems"}
    certain = {"definitely", "100%", "for sure", "absolutely", "clearly", "right now"}

    tone = "unknown"
    if any(h in text_lower for h in hedging):
        tone = "uncertain"
    elif any(c in text_lower for c in certain):
        tone = "confident"

    signal_type = "infrastructure" if high_matches > 0 else ("ambiguous" if medium_matches > 0 else "none")

    guidance = None
    if total_matches == 0:
        guidance = "No clear grid signals detected. Try mentioning what you see, hear, or what neighbors are saying."

    return {
        "valid": True,
        "signal_type": signal_type,
        "confidence_tone": tone,
        "keyword_matches": int(total_matches),
        "rejection_reason": None,
        "guidance": guidance
    }