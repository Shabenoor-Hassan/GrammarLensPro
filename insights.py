# [NEW] Readability metrics, CEFR level scoring, & pedagogical rules
import re
import math
import os
import json

from vocab_model import VocabUpgradeModel

try:
    from google import genai
    from google.genai import types as genai_types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False

# Your own fine-tuned local model — tried first in find_vocabulary_upgrades,
# before Gemini and the fixed dictionary. Lazy-loaded on first real use.
_local_vocab_model = VocabUpgradeModel()

# Pedagogical Grammar Explanations Database
GRAMMAR_RULES = {
    "Capitalization": {
        "rule": "Proper nouns and sentence beginnings must be capitalized.",
        "beginner_tip": "Always start every sentence with a capital letter and capitalize names, places, and the word 'I'.",
        "advanced_tip": "Consistent capitalization maintains professional authority and visual hierarchy in executive text."
    },
    "Grammar / Word Choice": {
        "rule": "Subject-Verb Agreement and Verb Tense Consistency.",
        "beginner_tip": "Make sure your verb matches when the action happened (past vs present) and who did it (singular vs plural).",
        "advanced_tip": "Maintain parallel verb structures across clauses to eliminate syntactic ambiguity."
    },
    "Redundant Word": {
        "rule": "Conciseness & Pleonasm Elimination.",
        "beginner_tip": "You used extra words that repeat the same meaning. Removing them makes your sentence punchy.",
        "advanced_tip": "Ruthlessly prune filler phrases to increase information density and executive impact."
    },
    "Missing Word / Punctuation": {
        "rule": "Syntactic Completeness and Flow.",
        "beginner_tip": "English requires articles ('a', 'an', 'the') or connecting words to link thoughts clearly.",
        "advanced_tip": "Precise punctuation guides reader pacing and eliminates comma splices."
    }
}

# Vocabulary Upgrade Dictionary — used as the fallback when the AI path
# (Gemini) isn't available, isn't configured, or fails for any reason.
VOCAB_UPGRADES = {
    "good": ["exceptional", "compelling", "commendable", "efficacious"],
    "bad": ["suboptimal", "detrimental", "unfavorable", "deficient"],
    "big": ["substantial", "extensive", "prominent", "considerable"],
    "make": ["generate", "fabricate", "establish", "formulate"],
    "help": ["facilitate", "assist", "expedite", "reinforce"],
    "show": ["demonstrate", "illustrate", "manifest", "substantiate"],
    "important": ["vital", "paramount", "critical", "imperative"],
    "change": ["transform", "modify", "refine", "recalibrate"],
    "get": ["acquire", "attain", "procure", "derive"],
    "problem": ["impediment", "quandary", "complication", "dilemma"]
}

# Below this many words, unique-words / total-words is too noisy to mean
# anything (a short sentence with no repeated word trivially hits 100%
# regardless of how varied the vocabulary actually is).
MIN_WORDS_FOR_VOCAB_RICHNESS = 12

# Check ai.google.dev/gemini-api/docs/models for the current free-tier
# lineup if this model name ever stops working — Google updates it over time.
_GEMINI_MODEL = "gemini-2.5-flash"
_GEMINI_TIMEOUT_MS = 8000  # best-effort; the try/except below is the real safety net either way

_gemini_client = None
_gemini_client_attempted = False


def _get_gemini_client():
    """Lazily creates the Gemini client on first use — same pattern as the
    local ExplanationGenerator's lazy model loading. No cost paid until the
    feature is actually needed, and it disables itself permanently for this
    process if it can't be set up (missing package, missing/invalid key)."""
    global _gemini_client, _gemini_client_attempted
    if _gemini_client_attempted:
        return _gemini_client
    _gemini_client_attempted = True

    if not _GENAI_AVAILABLE:
        print("⚠️ google-genai package not installed — vocabulary upgrades will use the rule-based dictionary only.")
        return None

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("⚠️ GEMINI_API_KEY not set — vocabulary upgrades will use the rule-based dictionary only.")
        return None

    try:
        _gemini_client = genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(timeout=_GEMINI_TIMEOUT_MS)
        )
    except Exception as e:
        print(f"⚠️ Could not create Gemini client ({e}) — falling back to the rule-based dictionary.")
        _gemini_client = None

    return _gemini_client


def count_syllables(word: str) -> int:
    """Estimates syllables in an English word."""
    word = word.lower()
    if len(word) <= 3:
        return 1
    word = re.sub(r'(?:[^laeiouy]|ed|es|e)$', '', word)
    word = re.sub(r'^y', '', word)
    matches = re.findall(r'[aeiouy]{1,2}', word)
    return max(1, len(matches))

def calculate_text_metrics(text: str):
    """Calculates Flesch Reading Ease, Grade Level, and estimated CEFR Level."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    num_sentences = max(1, len(sentences))

    words = re.findall(r'\b[A-Za-z]+\b', text)
    num_words = max(1, len(words))

    if num_words < 3:
        return {
            "reading_ease": 100,
            "cefr_level": "A1 (Beginner)",
            "vocab_richness": "N/A",
            "grade_level": "Elementary"
        }

    total_syllables = sum(count_syllables(w) for w in words)
    unique_words = set(w.lower() for w in words)

    if num_words < MIN_WORDS_FOR_VOCAB_RICHNESS:
        vocab_richness_display = "N/A (too short)"
    else:
        vocab_richness = round((len(unique_words) / num_words) * 100, 1)
        vocab_richness_display = f"{vocab_richness}%"

    # Flesch Reading Ease Formula
    fre = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (total_syllables / num_words)
    fre = max(0.0, min(100.0, fre))

    # Determine CEFR Level
    if fre >= 80:
        cefr = "A1 - A2 (Elementary)"
    elif fre >= 60:
        cefr = "B1 - B2 (Intermediate)"
    elif fre >= 40:
        cefr = "C1 (Advanced)"
    else:
        cefr = "C2 (Mastery / Academic)"

    return {
        "reading_ease": round(fre, 1),
        "cefr_level": cefr,
        "vocab_richness": vocab_richness_display,
        "word_count": num_words
    }


def _rule_based_vocabulary_upgrades(text: str):
    """The original fixed-dictionary version. Used whenever the AI path is
    unavailable or fails, so the feature never breaks — only 10 words are
    recognized, so most sentences will legitimately return nothing here."""
    words = re.findall(r'\b[A-Za-z]+\b', text.lower())
    suggestions = []
    for word in set(words):
        if word in VOCAB_UPGRADES:
            suggestions.append({
                "original": word,
                "alternatives": VOCAB_UPGRADES[word][:3],
                "tip": f"Elevate '{word}' to enhance nuance and professional impact."
            })
    return suggestions


def _ai_vocabulary_upgrades(text: str, user_level: str):
    """Asks Gemini to suggest upgrades tailored to the actual sentence,
    instead of only ever recognizing the same 10 hardcoded words. Returns
    None (not an empty list) if generation isn't usable for any reason, so
    the caller knows to fall back rather than reporting 'nothing found'."""
    client = _get_gemini_client()
    if client is None:
        return None

    prompt = (
        f"You are a vocabulary coach for a {user_level} English learner. "
        f"Given this sentence, identify up to 3 words that could be replaced "
        f"with more precise or sophisticated alternatives. Skip proper nouns "
        f"and words that are already advanced.\n\n"
        f"Sentence: \"{text.strip()}\"\n\n"
        f"Respond with ONLY a JSON array, no other text and no markdown "
        f"fences. Each item must have exactly these keys:\n"
        f'- "original": the exact word from the sentence, lowercase\n'
        f'- "alternatives": an array of 2-3 upgraded synonyms\n'
        f'- "tip": a one-sentence reason the upgrade helps\n\n'
        f"If nothing is worth upgrading, respond with: []"
    )

    try:
        response = client.models.generate_content(
            model=_GEMINI_MODEL,
            contents=prompt,
        )
        raw = (response.text or "").strip()

        # Defensive cleanup in case the model wraps the JSON in markdown fences anyway
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()

        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            return None

        cleaned = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            if "original" not in item or "alternatives" not in item:
                continue
            if not isinstance(item["alternatives"], list) or not item["alternatives"]:
                continue
            cleaned.append({
                "original": str(item["original"]),
                "alternatives": [str(a) for a in item["alternatives"][:3]],
                "tip": str(item.get("tip", "")) or f"Consider a stronger alternative to '{item['original']}'."
            })
        return cleaned

    except Exception as e:
        print(f"⚠️ Gemini vocabulary suggestion failed ({e}) — falling back to the rule-based dictionary.")
        return None


def find_vocabulary_upgrades(text: str, user_level: str = "intermediate"):
    """Finds words that can be upgraded to sound more professional. Tries
    your own fine-tuned local model first (no API, no network), then
    Gemini if that model isn't installed, then the fixed dictionary as a
    last resort — this feature never breaks the app no matter which of
    the three layers is or isn't available."""
    if user_level == "beginner":
        return []

    local_result = _local_vocab_model.find_upgrades(text)
    if local_result is not None:
        return local_result

    ai_result = _ai_vocabulary_upgrades(text, user_level)
    if ai_result is not None:
        return ai_result

    return _rule_based_vocabulary_upgrades(text)