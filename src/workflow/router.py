import re
import unicodedata

from pydantic import BaseModel

try:
    from langdetect import detect_langs
except ImportError:
    detect_langs = None


ISO_TO_LANGUAGE: dict[str, str] = {
    "fr": "french",
    "es": "spanish",
    "de": "german",
    "it": "italian",
    "pt": "portuguese",
    "nl": "dutch",
    "ru": "russian",
    "ar": "arabic",
    "ja": "japanese",
    "zh-cn": "chinese",
    "zh-tw": "chinese",
    "zh": "chinese",
    "ko": "korean",
    "hi": "hindi",
    "en": "english",
    "tr": "turkish",
    "pl": "polish",
    "sv": "swedish",
    "da": "danish",
    "fi": "finnish",
    "no": "norwegian",
    "cs": "czech",
    "el": "greek",
    "he": "hebrew",
    "id": "indonesian",
    "vi": "vietnamese",
    "th": "thai",
}

DISTINCTIVE_HINGLISH_WORDS = {
    "kardo", "kijiye", "karein", "karna", "karo", "chahiye", "wapas", "waapas", "rupaye",
    "katgaya", "katgaye", "katgayi", "batao", "batayein", "madad", "dikkat", "shukriya",
    "galti", "turant", "jaldi", "dijiye", "samadhan", "paise", "paisa", "rupay",
}

COMMON_HINGLISH_MARKERS = {
    "hai", "hain", "bhai", "yaar", "mera", "meri", "mere", "humne", "hamara", "kar",
    "hamari", "mujhe", "mujhko", "nahi", "nahin", "roko", "kholo",
    "raha", "rahi", "rahe", "kyun", "kaise", "kaisa", "aapka", "aapki", "aapke",
}


class LanguageScriptMetadata(BaseModel):
    script: str
    primary_language: str
    confidence: float
    is_supported_primary: bool


def detect_script(text: str) -> str:
    """Detect dominant Unicode script in the text (e.g. Latin, Devanagari, Arabic, CJK, etc.)."""
    if re.search(r"[\u0900-\u097F]", text):
        return "Devanagari"
    if re.search(r"[\u4E00-\u9FFF]", text):
        return "CJK"
    if re.search(r"[\u3040-\u30FF]", text):
        return "Kana"
    if re.search(r"[\u0600-\u06FF]", text):
        return "Arabic"
    if re.search(r"[\u0400-\u04FF]", text):
        return "Cyrillic"

    script_counts: dict[str, int] = {}
    for char in text:
        if char.isalpha():
            name = unicodedata.name(char, "")
            script = name.split()[0].title() if name else "Unknown"
            if "Latin" in script:
                script = "Latin"
            elif "Devanagari" in script:
                script = "Devanagari"
            elif "Arabic" in script:
                script = "Arabic"
            elif "Cjk" in script or "Ideograph" in script:
                script = "CJK"
            elif "Cyrillic" in script:
                script = "Cyrillic"
            script_counts[script] = script_counts.get(script, 0) + 1

    if not script_counts:
        return "Latin"

    return max(script_counts, key=script_counts.get)


def route_language_and_script(text: str) -> LanguageScriptMetadata:
    """Routes and classifies the language and script of an incoming ticket."""
    detected_script = detect_script(text)

    # 1. Direct Devanagari check -> Hindi
    if detected_script == "Devanagari":
        return LanguageScriptMetadata(
            script="Devanagari",
            primary_language="hindi",
            confidence=0.99,
            is_supported_primary=True,
        )

    # 2. Hinglish marker check (Latin script code-mixed Hindi)
    tokens = set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))
    if tokens.intersection(DISTINCTIVE_HINGLISH_WORDS) or len(tokens.intersection(COMMON_HINGLISH_MARKERS)) >= 2:
        return LanguageScriptMetadata(
            script="Latin",
            primary_language="hinglish",
            confidence=0.95,
            is_supported_primary=True,
        )

    # 3. ISO language detection
    lang = "english"
    conf = 0.95

    if detect_langs:
        try:
            langs = detect_langs(text)
            if langs:
                iso_code = langs[0].lang
                lang = ISO_TO_LANGUAGE.get(iso_code, iso_code)
                conf = round(langs[0].prob, 3)
        except Exception:
            lang = "english"
            conf = 0.5
    else:
        lang = "english"
        conf = 0.9

    is_primary = lang in ["english", "spanish", "french", "german", "hindi", "hinglish"]

    return LanguageScriptMetadata(
        script=detected_script,
        primary_language=lang,
        confidence=conf,
        is_supported_primary=is_primary,
    )
