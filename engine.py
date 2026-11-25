import difflib
import re
import os
import string
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

try:
    from google import genai
    from google.genai import types as genai_types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False

# Fix for Intel 12th Gen CPU thread deadlock
torch.set_num_threads(4)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

# ==========================================================================
# GRAMMAR KNOWLEDGE TABLES (used for richer, more specific classification)
# ==========================================================================

ARTICLES = {"a", "an", "the"}

PREPOSITIONS = {
    "in", "on", "at", "to", "for", "with", "by", "from", "of", "about",
    "into", "onto", "over", "under", "between", "among", "through",
    "during", "before", "after", "since", "until", "against", "toward",
    "towards", "within", "without", "across", "along", "around"
}

BE_VERB_FORMS = {"is", "am", "are", "was", "were", "be", "been", "being"}

# Groups of related forms of the same irregular verb (base, present, -ing, past, past participle)
VERB_FAMILIES = [
    {"go", "goes", "going", "went", "gone"},
    {"buy", "buys", "buying", "bought"},
    {"eat", "eats", "eating", "ate", "eaten"},
    {"see", "sees", "seeing", "saw", "seen"},
    {"do", "does", "doing", "did", "done"},
    {"have", "has", "having", "had"},
    BE_VERB_FORMS,
    {"write", "writes", "writing", "wrote", "written"},
    {"run", "runs", "running", "ran"},
    {"come", "comes", "coming", "came"},
    {"take", "takes", "taking", "took", "taken"},
    {"make", "makes", "making", "made"},
    {"give", "gives", "giving", "gave", "given"},
    {"know", "knows", "knowing", "knew", "known"},
    {"think", "thinks", "thinking", "thought"},
    {"get", "gets", "getting", "got", "gotten"},
    {"say", "says", "saying", "said"},
    {"speak", "speaks", "speaking", "spoke", "spoken"},
    {"drive", "drives", "driving", "drove", "driven"},
    {"read", "reads", "reading"},
    {"want", "wants", "wanting", "wanted"},
]

# ==========================================================================
# POST-PROCESSING RULES — patches known weaknesses of the local T5 model
# ==========================================================================

# 1. Day & Month capitalization
_DAYS = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
_MONTHS = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december"
}
_PROPER_NOUNS_LOWER = _DAYS | _MONTHS

# 2. Phonetic a/an — words that START with a vowel LETTER but a CONSONANT SOUND
#    (e.g. "university" starts with /juː/, so it takes "a", not "an")
_AN_EXCEPTIONS_CONSONANT_SOUND = {
    "united", "university", "uniform", "union", "unique", "unit",
    "universal", "usage", "useful", "useless", "user", "usual",
    "usually", "utensil", "utility", "utilize", "utopia", "unanimous",
    "unicorn", "unilateral", "uranium", "urinal", "usurp", "uterus",
    "euphemism", "euphoria", "european", "eucalyptus", "eulogy",
    "ewe", "one", "once",
}
#    Words that START with a consonant LETTER but a VOWEL SOUND
#    (e.g. "honest" starts with /ɒ/, so it takes "an", not "a")
_A_EXCEPTIONS_VOWEL_SOUND = {
    "honest", "honestly", "honesty", "honor", "honour", "honorable",
    "honourable", "hour", "hourly", "heir", "heiress", "herb",
    "herbalist", "herbal",
}

# 3. Irregular plurals — mapping singular → correct plural
_IRREGULAR_PLURALS = {
    "loaf": "loaves", "leaf": "leaves", "knife": "knives",
    "wife": "wives", "life": "lives", "half": "halves",
    "calf": "calves", "wolf": "wolves", "shelf": "shelves",
    "thief": "thieves", "self": "selves", "elf": "elves",
    "child": "children", "man": "men", "woman": "women",
    "tooth": "teeth", "foot": "feet", "goose": "geese",
    "mouse": "mice", "louse": "lice", "ox": "oxen",
    "person": "people", "cactus": "cacti", "focus": "foci",
    "fungus": "fungi", "nucleus": "nuclei", "syllabus": "syllabi",
    "analysis": "analyses", "basis": "bases", "crisis": "crises",
    "diagnosis": "diagnoses", "thesis": "theses",
    "phenomenon": "phenomena", "criterion": "criteria",
    "datum": "data", "medium": "media",
}
# Also build the malformed plural → correct plural lookup
# e.g. "loafs" → "loaves", "childs" → "children"
_BAD_PLURAL_TO_CORRECT = {}
for _sing, _plur in _IRREGULAR_PLURALS.items():
    _BAD_PLURAL_TO_CORRECT[_sing + "s"] = _plur
    _BAD_PLURAL_TO_CORRECT[_sing + "es"] = _plur

# Determiners that signal the following noun should be plural
_PLURAL_DETERMINERS = {
    "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "twenty", "thirty", "hundred", "thousand",
    "many", "few", "several", "various", "numerous", "multiple",
    "some", "these", "those", "all", "both",
}

# 4. Present Perfect + past-time anchor conflict
#    "has/have/had + past-participle + ... + <past time marker>" → simple past
_PAST_TIME_MARKERS = {
    "yesterday", "ago", "last week", "last month", "last year",
    "last night", "last monday", "last tuesday", "last wednesday",
    "last thursday", "last friday", "last saturday", "last sunday",
    "in 1990", "in 2000", "in 2010", "in 2020",
}

# Past participle → simple past mapping for common irregular verbs
_PARTICIPLE_TO_PAST = {
    "arrived": "arrived", "gone": "went", "been": "was",
    "eaten": "ate", "seen": "saw", "done": "did",
    "written": "wrote", "taken": "took", "given": "gave",
    "known": "knew", "spoken": "spoke", "driven": "drove",
    "chosen": "chose", "broken": "broke", "stolen": "stole",
    "forgotten": "forgot", "begun": "began", "drunk": "drank",
    "sung": "sang", "swum": "swam", "rung": "rang",
    "bought": "bought", "brought": "brought", "built": "built",
    "caught": "caught", "felt": "felt", "found": "found",
    "got": "got", "gotten": "got", "had": "had",
    "heard": "heard", "held": "held", "kept": "kept",
    "left": "left", "lost": "lost", "made": "made",
    "met": "met", "paid": "paid", "read": "read",
    "said": "said", "sent": "sent", "sat": "sat",
    "sold": "sold", "spent": "spent", "stood": "stood",
    "thought": "thought", "told": "told", "understood": "understood",
    "won": "won", "wore": "wore", "worn": "wore",
}


def _post_process_corrections(text: str) -> str:
    """Rule-based post-processing that patches known blind spots of the
    local vennify/t5-base model. Runs after model inference, before
    error detection, so the diff picks up these fixes too."""
    text = _fix_proper_noun_capitalization(text)
    text = _fix_phonetic_articles(text)
    text = _fix_irregular_plurals(text)
    text = _fix_perfect_tense_conflict(text)
    return text


def _fix_proper_noun_capitalization(text: str) -> str:
    """Capitalizes days of the week and month names."""
    words = text.split()
    for i, w in enumerate(words):
        # Strip trailing punctuation for comparison, but preserve it
        core = w.rstrip(".,!?;:'\"-)")
        tail = w[len(core):]
        if core.lower() in _PROPER_NOUNS_LOWER:
            words[i] = core.capitalize() + tail
    return " ".join(words)


def _fix_phonetic_articles(text: str) -> str:
    """Fixes a/an based on the SOUND of the following word, not just
    the letter — e.g. 'an university' → 'a university',
    'a honest man' → 'an honest man'."""
    words = text.split()
    for i in range(len(words) - 1):
        art_core = words[i].rstrip(".,!?;:'\"-)")
        art_lower = art_core.lower()
        if art_lower not in ("a", "an"):
            continue

        next_core = words[i + 1].rstrip(".,!?;:'\"-)")
        next_lower = next_core.lower()
        if not next_lower:
            continue

        # Decide what the article SHOULD be
        if next_lower in _AN_EXCEPTIONS_CONSONANT_SOUND:
            correct_art = "a"
        elif next_lower in _A_EXCEPTIONS_VOWEL_SOUND:
            correct_art = "an"
        elif next_lower[0] in "aeiou":
            correct_art = "an"
        else:
            correct_art = "a"

        if art_lower != correct_art:
            # Preserve original capitalization style
            if art_core[0].isupper():
                correct_art = correct_art.capitalize()
            tail = words[i][len(art_core):]
            words[i] = correct_art + tail

    return " ".join(words)


def _fix_irregular_plurals(text: str) -> str:
    """After a plural determiner ('two', 'many', 'several', etc.) fixes
    a singular irregular noun to its correct plural form.
    Also fixes malformed regular-style plurals like 'loafs' → 'loaves'."""
    words = text.split()
    for i in range(len(words)):
        core = words[i].rstrip(".,!?;:'\"-)")
        tail = words[i][len(core):]
        core_lower = core.lower()

        # Fix malformed plurals anywhere (e.g. "loafs" → "loaves")
        if core_lower in _BAD_PLURAL_TO_CORRECT:
            replacement = _BAD_PLURAL_TO_CORRECT[core_lower]
            # Preserve capitalization
            if core[0].isupper():
                replacement = replacement.capitalize()
            words[i] = replacement + tail
            continue

        # After a plural determiner, fix singular → plural
        if i > 0:
            prev_core = words[i - 1].rstrip(".,!?;:'\"-)")
            if prev_core.lower() in _PLURAL_DETERMINERS and core_lower in _IRREGULAR_PLURALS:
                replacement = _IRREGULAR_PLURALS[core_lower]
                if core[0].isupper():
                    replacement = replacement.capitalize()
                words[i] = replacement + tail

    return " ".join(words)


def _fix_perfect_tense_conflict(text: str) -> str:
    """Detects 'has/have + past-participle ... <past time marker>' and
    rewrites to simple past — e.g. 'He has arrived ... two days ago'
    → 'He arrived ... two days ago'."""
    text_lower = text.lower()

    # Only fire if there's a past-time marker present
    has_marker = False
    for marker in _PAST_TIME_MARKERS:
        if marker in text_lower:
            has_marker = True
            break
    if not has_marker:
        return text

    # Look for "has/have/had <participle>" pattern
    pattern = re.compile(
        r'\b(has|have|had)\s+(\w+)\b',
        re.IGNORECASE
    )

    def _replace_perfect(match):
        aux = match.group(1)
        participle = match.group(2)
        part_lower = participle.lower()

        if part_lower in _PARTICIPLE_TO_PAST:
            simple_past = _PARTICIPLE_TO_PAST[part_lower]
            # Preserve capitalization of the participle
            if participle[0].isupper():
                simple_past = simple_past.capitalize()
            return simple_past
        # Regular verb past participle ending in -ed is already simple past
        if part_lower.endswith("ed") and part_lower not in ("need", "seed", "feed", "bleed"):
            return participle
        # Can't determine simple past form — leave unchanged
        return match.group(0)

    return pattern.sub(_replace_perfect, text)


PUNCTUATION_CHARS = set(".,!?;:'\"-()")


def _same_verb_family(a: str, b: str) -> bool:
    a, b = a.lower(), b.lower()
    for fam in VERB_FAMILIES:
        if a in fam and b in fam:
            return True
    return False


def classify_error(orig_chunk: str, corr_chunk: str, tag: str) -> str:
    """Assigns a specific grammatical category to a detected difference,
    instead of lumping everything that isn't a delete/insert/case-change
    into one generic 'Grammar / Word Choice' bucket."""
    orig = orig_chunk.strip()
    corr = corr_chunk.strip()
    orig_lower = orig.lower()
    corr_lower = corr.lower()

    if tag == "delete":
        if orig and all(ch in PUNCTUATION_CHARS for ch in orig):
            return "Punctuation"
        return "Redundant Word"

    if tag == "insert":
        if corr and all(ch in PUNCTUATION_CHARS for ch in corr):
            return "Punctuation"
        return "Missing Word"

    if tag == "replace":
        orig_words = orig.split()
        corr_words = corr.split()

        # Multi-word replacements are sentence-structure issues, not single-word ones
        if len(orig_words) > 1 or len(corr_words) > 1:
            return "Word Order / Syntax"

        if not orig_lower or not corr_lower:
            return "Syntax"

        # Strip trailing punctuation before comparing the "core" word. This
        # matters most for very short/single-word inputs: with nothing else
        # for the differ to anchor on, a case change and an added period can
        # get merged into one "replace" chunk (e.g. "school" -> "School."),
        # which would otherwise fall through every category below and get
        # mislabeled as Spelling just because the strings look similar.
        orig_core = orig.rstrip(string.punctuation)
        corr_core = corr.rstrip(string.punctuation)
        if orig_core and orig_core.lower() == corr_core.lower():
            if orig_core != corr_core:
                return "Capitalization"
            return "Punctuation"

        # Articles
        if orig_lower in ARTICLES or corr_lower in ARTICLES:
            return "Article Usage"

        # Prepositions
        if orig_lower in PREPOSITIONS and corr_lower in PREPOSITIONS:
            return "Preposition"

        # Verb tense / subject-verb agreement (same verb family, different form)
        if _same_verb_family(orig_lower, corr_lower):
            if orig_lower in BE_VERB_FORMS or corr_lower in BE_VERB_FORMS:
                return "Subject-Verb Agreement"
            return "Verb Tense"

        # Simple pluralization (word vs word+s/es)
        if (
            corr_lower == orig_lower + "s" or corr_lower == orig_lower + "es"
            or orig_lower == corr_lower + "s" or orig_lower == corr_lower + "es"
        ):
            return "Pluralization"

        # Spelling: very similar strings that aren't identical
        similarity = difflib.SequenceMatcher(None, orig_lower, corr_lower).ratio()
        if similarity > 0.6 and abs(len(orig_lower) - len(corr_lower)) <= 3:
            return "Spelling"

        return "Word Choice"

    return "Syntax"


# Per-category rule text + a genuinely distinct explanation for each of the
# three target levels, built from the actual words involved rather than one
# fixed sentence for the whole category.
_EXPLANATION_TEMPLATES = {
    "Capitalization": {
        "rule": "Proper nouns and the first word of a sentence must be capitalized.",
        "beginner": "Always start a sentence with a capital letter. Change '{o}' to '{r}'.",
        "intermediate": "'{o}' should be capitalized as '{r}' since it starts a sentence or is a proper noun.",
        "advanced": "Capitalization convention requires '{r}' rather than '{o}' for sentence-initial words and proper nouns.",
    },
    "Verb Tense": {
        "rule": "The verb tense must match the time frame described in the sentence.",
        "beginner": "Use '{r}' instead of '{o}' to show when the action happened.",
        "intermediate": "'{o}' is the wrong tense here — '{r}' correctly matches the sentence's time frame.",
        "advanced": "'{o}' does not align with the established temporal context; '{r}' is the tense-consistent form.",
    },
    "Subject-Verb Agreement": {
        "rule": "The verb must agree in number and person with its subject.",
        "beginner": "Change '{o}' to '{r}' so the verb matches the subject.",
        "intermediate": "'{o}' doesn't agree with the subject — '{r}' is the correct matching form.",
        "advanced": "Subject-verb agreement requires '{r}' rather than '{o}' to reflect the subject's person and number.",
    },
    "Article Usage": {
        "rule": "Articles ('a', 'an', 'the') must match whether a noun is specific, general, or countable.",
        "beginner": "Use '{r}' instead of '{o}' before this word.",
        "intermediate": "'{r}' is the correct article here instead of '{o}', based on how specific the noun is.",
        "advanced": "Article choice here depends on definiteness and countability; '{r}' is correct in place of '{o}'.",
    },
    "Preposition": {
        "rule": "Certain verbs, adjectives, and nouns pair with specific prepositions by convention.",
        "beginner": "Use '{r}' instead of '{o}' here.",
        "intermediate": "'{o}' isn't the preposition normally used in this context — '{r}' is the natural choice.",
        "advanced": "Idiomatic collocation requires '{r}' rather than '{o}' in this construction.",
    },
    "Pluralization": {
        "rule": "Nouns must be singular or plural depending on the quantity referred to.",
        "beginner": "Change '{o}' to '{r}' to show the correct number.",
        "intermediate": "'{o}' should be '{r}' to correctly reflect singular vs. plural.",
        "advanced": "Number agreement requires '{r}' rather than '{o}' given the referenced quantity.",
    },
    "Spelling": {
        "rule": "Words must be spelled correctly according to standard English orthography.",
        "beginner": "'{o}' is spelled incorrectly. The correct spelling is '{r}'.",
        "intermediate": "'{o}' is a spelling error — it should be written as '{r}'.",
        "advanced": "'{o}' deviates from the standard orthographic form '{r}'.",
    },
    "Word Choice": {
        "rule": "Word choice should fit the meaning and register of the sentence.",
        "beginner": "'{r}' is a better word to use here instead of '{o}'.",
        "intermediate": "'{r}' fits the sentence's meaning better than '{o}' in this context.",
        "advanced": "'{r}' is more precise or idiomatic than '{o}' in this context.",
    },
    "Punctuation": {
        "rule": "Punctuation marks structure and separate ideas within a sentence.",
        "beginner": "Add or fix the punctuation: '{r}'.",
        "intermediate": "Punctuation is needed here — '{r}' completes the sentence correctly.",
        "advanced": "Correct punctuation ('{r}') is required here to properly delimit the clause.",
    },
    "Missing Word": {
        "rule": "A necessary word was omitted from the sentence.",
        "beginner": "You're missing a word. Add '{r}'.",
        "intermediate": "This sentence needs '{r}' to be grammatically complete.",
        "advanced": "The sentence is missing '{r}', which is required for grammatical completeness.",
    },
    "Redundant Word": {
        "rule": "Unnecessary words should be removed for clarity and conciseness.",
        "beginner": "You don't need '{o}' here. Remove it.",
        "intermediate": "'{o}' is unnecessary and can be removed for a cleaner sentence.",
        "advanced": "'{o}' is redundant here; removing it improves concision without changing meaning.",
    },
    "Word Order / Syntax": {
        "rule": "Word order affects clarity and grammatical correctness in English sentence structure.",
        "beginner": "Try saying it this way instead: '{r}'.",
        "intermediate": "Reordering to '{r}' makes the sentence clearer and grammatically correct.",
        "advanced": "The original word order ('{o}') is non-standard; '{r}' follows conventional English syntax.",
    },
    "Syntax": {
        "rule": "Standard English grammatical convention applies here.",
        "beginner": "Use '{r}' instead of '{o}'.",
        "intermediate": "'{r}' is the grammatically correct form here rather than '{o}'.",
        "advanced": "'{r}' aligns with standard grammatical structure, unlike '{o}'.",
    },
}


def explain_error(error_type: str, original: str, replacement: str, level: str):
    """Builds a rule description and a level-appropriate tip that reference
    the actual words involved, instead of returning one fixed sentence."""
    level = (level or "intermediate").lower()
    if level not in ("beginner", "intermediate", "advanced"):
        level = "intermediate"

    o = original.strip() or "(nothing)"
    r = replacement.strip() or "(removed)"

    entry = _EXPLANATION_TEMPLATES.get(error_type, _EXPLANATION_TEMPLATES["Syntax"])
    tip = entry[level].format(o=o, r=r)
    return entry["rule"], tip


class ExplanationGenerator:
    """Generates natural-language pedagogical explanations using the Gemini API
    instead of loading a second heavy local transformer model into RAM.

    This preserves vennify/t5-base-grammar-correction as the SINGLE local Seq2Seq
    model for sentence correction, while offloading rich pedagogical explanations
    to the Gemini API on-demand.

    Graceful degradation:
      If GEMINI_API_KEY is not set, network is offline, or the API fails, it
      silently returns None so the application seamlessly falls back to the
      rule-based templates from explain_error().
    """

    MIN_ACCEPTABLE_LENGTH = 10

    def __init__(self, model_name: str = "gemini-2.5-flash", enabled: bool = True):
        self.model_name = model_name
        self.enabled = enabled
        self._client = None
        self._client_attempted = False

    def _ensure_client(self):
        """Initializes the Gemini API client lazily on first actual explanation request."""
        if self._client_attempted:
            return self._client
        self._client_attempted = True

        if not self.enabled:
            return None

        if not _GENAI_AVAILABLE:
            print("[WARNING] google-genai package not installed - falling back to rule-based explanations.")
            self.enabled = False
            return None

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("[WARNING] GEMINI_API_KEY not found in environment - falling back to rule-based explanations.")
            self.enabled = False
            return None

        try:
            self._client = genai.Client(
                api_key=api_key,
                http_options=genai_types.HttpOptions(timeout=15000)
            )
            print(f"[INFO] Gemini API explanation client initialized ({self.model_name})")
        except Exception as e:
            print(f"[WARNING] Could not initialize Gemini client ({e}) - falling back to rule-based explanations.")
            self._client = None
            self.enabled = False

        return self._client

    def generate(self, error_type: str, original: str, replacement: str, sentence: str, level: str):
        """Returns a concise, pedagogical explanation string via Gemini API,
        or None if generation is unavailable (signaling callers to use the rule-based fallback)."""
        client = self._ensure_client()
        if not client:
            return None

        o = original.strip()
        r = replacement.strip()
        clean_sentence = sentence.strip()

        # Build specific context instruction based on mutation type
        if not o and r:
            action_desc = f"why the missing word/punctuation '{r}' should be added to this sentence: \"{clean_sentence}\""
        elif o and not r:
            action_desc = f"why the redundant word '{o}' should be removed from this sentence: \"{clean_sentence}\""
        else:
            action_desc = f"why '{o}' should be corrected to '{r}' in this sentence: \"{clean_sentence}\""

        prompt = (
            f"You are a friendly and encouraging English grammar tutor. "
            f"Explain in ONE concise sentence (15 to 25 words maximum) for a {level} English learner "
            f"{action_desc}. The mistake type is '{error_type}'. "
            f"Do NOT repeat the entire sentence. Give only the single clear teaching sentence."
        )

        try:
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=200,
                    thinking_config=genai_types.ThinkingConfig(thinking_budget=0)
                )
            )
            text = response.text.strip() if response and response.text else ""
            if len(text) < self.MIN_ACCEPTABLE_LENGTH:
                return None
            return text
        except Exception:
            # On any API/network exception, gracefully degrade to None so fallback template is used
            return None


class GrammarEngine:
    def __init__(self, model_name: str = "vennify/t5-base-grammar-correction"):
        print(f"Loading tokenizer: {model_name}...")
        self.device = "cpu"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                low_cpu_mem_usage=True,
                local_files_only=True
            ).to(self.device)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                low_cpu_mem_usage=True
            ).to(self.device)

        self.model.eval()
        print("[INFO] Model loaded successfully on CPU!")

    def correct_sentence(self, text: str) -> str:
        """Runs the transformer to generate a corrected sentence."""
        cleaned = text.strip()
        if not cleaned:
            return ""

        input_text = f"gec: {cleaned}"
        inputs = self.tokenizer(
            input_text,
            padding="longest",
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(self.device)

        # Cap output length relative to the input instead of a flat 256 tokens.
        # A short input has no legitimate reason to produce a long output —
        # giving beam search that much unused room is exactly what lets it
        # pad the result out, whether by repeating itself or, for very short
        # fragments, inventing new content to "complete" the sentence.
        input_len = inputs["input_ids"].shape[-1]
        dynamic_max_length = min(256, max(10, input_len * 3))

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=dynamic_max_length,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,  # blocks the model from re-emitting a 3+ token chunk it already generated
                repetition_penalty=1.3,  # softly discourages repeating earlier tokens in general
            )

        corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        corrected = self._dedupe_repeated_sentence(corrected)

        # Tightening max_length above reduces the room for fabrication, but
        # doesn't eliminate the model's tendency to "complete" a bare word or
        # fragment into a full invented sentence — it can still do that in
        # very few tokens. This catches that case directly: if the output
        # grew far more than a genuine grammar fix ever would for such a
        # short input, treat it as fabricated and use a minimal, honest
        # correction instead of inventing nothing at all.
        if self._looks_fabricated(cleaned, corrected):
            corrected = self._minimal_safe_correction(cleaned)

        # Post-processing rules: patch known blind spots of the local T5
        # model (irregular plurals, phonetic articles, day/month caps,
        # present-perfect + past-time conflicts).
        corrected = _post_process_corrections(corrected)

        return corrected

    def _looks_fabricated(self, original: str, corrected: str) -> bool:
        """Flags cases where a short input's word count grew far more than a
        real grammar correction would — a strong signal the model invented
        new content rather than just fixing what was there."""
        orig_words = re.findall(r"[A-Za-z']+", original)
        corr_words = re.findall(r"[A-Za-z']+", corrected)
        if not orig_words:
            return False

        growth = len(corr_words) / len(orig_words)
        return len(orig_words) <= 3 and growth > 2.5

    def _minimal_safe_correction(self, text: str) -> str:
        """A fallback that only ever does mechanical, safe fixes — capitalize
        the first letter and ensure terminal punctuation — and never invents
        a single new word. Used when the model's output looks fabricated."""
        cleaned = text.strip()
        if not cleaned:
            return cleaned
        cleaned = cleaned[0].upper() + cleaned[1:]
        if cleaned[-1] not in ".!?":
            cleaned += "."
        return cleaned

    def _dedupe_repeated_sentence(self, text: str) -> str:
        """Defensive safety net: if the model still outputs a sentence
        immediately followed by a near-duplicate of itself, keep only the
        first occurrence. This runs regardless of whether the generation
        settings above fully prevented it, so the UI is protected either
        way instead of depending on generation tuning alone."""
        if not text:
            return text

        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if len(sentences) < 2:
            return text

        deduped = [sentences[0]]
        for s in sentences[1:]:
            prev = deduped[-1].strip().lower().rstrip(".!?")
            cur = s.strip().lower().rstrip(".!?")
            if cur and cur == prev:
                continue  # skip an exact repeat of the sentence right before it
            deduped.append(s)

        return " ".join(deduped)

    def detect_errors(self, original_text: str, corrected_text: str, user_level: str = "intermediate", explainer=None):
        """Aligns original text and corrected text to pinpoint errors, classifies
        each one into a specific grammatical category, and attaches an explanation.

        If an ExplanationGenerator is passed in via `explainer`, each tip is
        generated fresh from the actual sentence; if generation is unavailable
        or produces something unusable, it falls back to the rule-based
        template automatically — every error always gets a tip either way."""
        # Tokenize words, punctuation, and whitespace as separate tokens so that
        # e.g. "school" -> "school." is detected as an inserted period (Punctuation),
        # not a near-identical "Spelling" change on the word itself.
        orig_words = [m.group(0) for m in re.finditer(r'\w+|[^\w\s]|\s+', original_text)]
        corr_words = [m.group(0) for m in re.finditer(r'\w+|[^\w\s]|\s+', corrected_text)]

        matcher = difflib.SequenceMatcher(None, orig_words, corr_words)
        errors = []

        orig_char_indices = []
        curr = 0
        for w in orig_words:
            orig_char_indices.append((curr, curr + len(w)))
            curr += len(w)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                continue

            orig_chunk = "".join(orig_words[i1:i2])
            corr_chunk = "".join(corr_words[j1:j2])

            if orig_chunk.strip() == "" and corr_chunk.strip() == "":
                continue

            start_char = orig_char_indices[i1][0] if i1 < len(orig_char_indices) else len(original_text)
            end_char = orig_char_indices[i2 - 1][1] if i2 > 0 and (i2 - 1) < len(orig_char_indices) else len(original_text)

            error_type = classify_error(orig_chunk, corr_chunk, tag)

            if tag == 'delete':
                description = f"Consider removing '{orig_chunk.strip()}'"
            elif tag == 'insert':
                description = f"Insert '{corr_chunk.strip()}'"
            else:
                description = f"Replace with '{corr_chunk.strip()}'"

            grammar_rule, pedagogical_tip = explain_error(error_type, orig_chunk, corr_chunk, user_level)
            tip_source = "rule"

            if explainer is not None:
                generated_tip = explainer.generate(error_type, orig_chunk, corr_chunk, original_text, user_level)
                if generated_tip:
                    pedagogical_tip = generated_tip
                    tip_source = "ai"

            errors.append({
                "id": len(errors) + 1,
                "type": error_type,
                "description": description,
                "original": orig_chunk,
                "replacement": corr_chunk,
                "start": start_char,
                "end": end_char,
                "grammar_rule": grammar_rule,
                "pedagogical_tip": pedagogical_tip,
                "tip_source": tip_source,
            })

        return errors


# Standalone test when running engine.py directly
if __name__ == "__main__":
    eng = GrammarEngine()
    explainer = ExplanationGenerator()  # uses Gemini API; auto-falls back to rules if unavailable
    test_sentence = "He are going to school yesterday and buyed a apple."
    print(f"\nOriginal : {test_sentence}")
    corrected = eng.correct_sentence(test_sentence)
    print(f"Corrected: {corrected}")
    for lvl in ("beginner", "intermediate", "advanced"):
        print(f"\n--- {lvl} ---")
        for err in eng.detect_errors(test_sentence, corrected, lvl, explainer=explainer):
            print(f"[{err['tip_source']}] {err['type']} -> {err['pedagogical_tip']}")