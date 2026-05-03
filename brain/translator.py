from typing import Tuple

SUPPORTED_LANGUAGES = {
    "en": "English", "sw": "Swahili", "ar": "Arabic", "fr": "French",
    "es": "Spanish", "pt": "Portuguese", "ur": "Urdu", "hi": "Hindi",
    "tl": "Filipino", "yo": "Yoruba", "ha": "Hausa", "ig": "Igbo",
    "am": "Amharic", "zh": "Chinese", "id": "Indonesian"
}

def detect_and_translate(text: str) -> Tuple[str, str, str]:
    try:
        from langdetect import detect
        detected_lang = detect(text)
    except Exception:
        detected_lang = "en"

    if detected_lang == "en" or detected_lang not in SUPPORTED_LANGUAGES:
        return text, detected_lang, SUPPORTED_LANGUAGES.get(detected_lang, "Unknown")

    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source=detected_lang, target="en")
        translated = translator.translate(text)
        lang_name = SUPPORTED_LANGUAGES.get(detected_lang, detected_lang)
        return translated, detected_lang, lang_name
    except Exception:
        return text, detected_lang, SUPPORTED_LANGUAGES.get(detected_lang, "Unknown")

def translate_output(text: str, target_language: str) -> str:
    if target_language == "en":
        return text
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source="en", target=target_language)
        return translator.translate(text)
    except Exception:
        return text