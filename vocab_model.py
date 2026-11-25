"""
Local inference wrapper for the fine-tuned vocabulary-upgrade model.

This is the payoff of the distillation pipeline: Gemini was used ONCE,
offline, to build a training set (generate_training_data.py); a t5-small
was fine-tuned on it in Colab; and this module runs that fine-tuned model
locally. The shipped app needs no API key and no network for this feature.

Follows the same two principles as the rest of the app:
  - Lazy loading: weights are only read into RAM on first actual use, so
    an 8GB laptop doesn't pay for a feature nobody clicked.
  - Graceful degradation: if the model folder is missing or anything fails,
    every method returns None and the caller falls back to its own logic.
"""

import os
import re
import json

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

# Must stay in sync with CANDIDATE_WORDS in generate_training_data.py —
# the model was only ever trained on these words, so asking it about
# anything else produces unreliable output.
# The word list the model was actually trained on. Loaded from
# vocab_dictionary.json (built by build_vocab_dictionary.py) at import
# time, so this never has to be manually kept in sync with a second copy —
# there's only one source of truth now. Falls back to a small built-in
# list so the app still runs even before that script has ever been run.
_DICTIONARY_PATH = "vocab_dictionary.json"
_FALLBACK_CANDIDATE_WORDS = {
    "good", "bad", "big", "important", "help", "show", "change", "problem", "nice", "happy",
}


def _load_candidate_words():
    if os.path.exists(_DICTIONARY_PATH):
        try:
            with open(_DICTIONARY_PATH, "r", encoding="utf-8") as f:
                words = set(json.load(f).keys())
            if words:
                return words
        except Exception as e:
            print(f"⚠️ Could not read {_DICTIONARY_PATH} ({e}) — using the small built-in word list instead.")
    else:
        print(f"⚠️ {_DICTIONARY_PATH} not found — using the small built-in word list instead. Run build_vocab_dictionary.py to generate it.")
    return _FALLBACK_CANDIDATE_WORDS


CANDIDATE_WORDS = _load_candidate_words()

DEFAULT_MODEL_DIR = "vocab_upgrade_model"
MAX_WORDS_PER_SENTENCE = 3  # don't overwhelm the user with suggestions


class VocabUpgradeModel:
    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR, enabled: bool = True):
        self.model_dir = model_dir
        self.enabled = enabled
        self.model = None
        self.tokenizer = None
        self._load_attempted = False

    def _ensure_loaded(self):
        """Loads the fine-tuned weights on first use only."""
        if self._load_attempted:
            return
        self._load_attempted = True

        if not self.enabled:
            return

        if not _TRANSFORMERS_AVAILABLE:
            print("⚠️ transformers not available — vocabulary upgrades will use the fallback.")
            self.enabled = False
            return

        if not os.path.isdir(self.model_dir):
            print(
                f"⚠️ Fine-tuned model folder '{self.model_dir}' not found — "
                f"vocabulary upgrades will use the fallback. "
                f"(Unzip vocab_upgrade_model.zip into your project folder to enable it.)"
            )
            self.enabled = False
            return

        try:
            print(f"Loading fine-tuned vocabulary model from '{self.model_dir}'... (first use)")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_dir).to("cpu")
            self.model.eval()
            print("✅ Fine-tuned vocabulary model loaded on CPU!")
        except Exception as e:
            print(f"⚠️ Could not load fine-tuned vocabulary model ({e}) — using the fallback.")
            self.enabled = False
            self.model = None
            self.tokenizer = None

    def _suggest_for_word(self, word: str, sentence: str):
        """Returns a list of alternatives for one word, or None."""
        # This prompt shape must match the notebook's input_text format exactly.
        prompt = f"upgrade: {word} context: {sentence.strip()}"
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=96)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=32,
                    num_beams=4,
                    early_stopping=True,
                )
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        except Exception:
            return None

        if not text:
            return None

        alts = [a.strip() for a in text.split(",")]
        # Drop empties, duplicates, and any "upgrade" identical to the original word
        seen = set()
        cleaned = []
        for a in alts:
            low = a.lower()
            if not a or low == word.lower() or low in seen:
                continue
            seen.add(low)
            cleaned.append(a)

        return cleaned[:3] or None

    def find_upgrades(self, text: str):
        """Returns a list of upgrade suggestions in the same shape the app's
        frontend already expects, or None if the model isn't usable — the
        caller should treat None as 'fall back to the dictionary'."""
        self._ensure_loaded()
        if not self.enabled or self.model is None:
            return None

        words = re.findall(r"[A-Za-z']+", text.lower())

        # Only ask about words the model was actually trained on, preserving
        # order of first appearance and skipping repeats.
        targets = []
        seen = set()
        for w in words:
            if w in CANDIDATE_WORDS and w not in seen:
                seen.add(w)
                targets.append(w)
            if len(targets) >= MAX_WORDS_PER_SENTENCE:
                break

        if not targets:
            # None of the sentence's words are in this model's fixed 95-word
            # training vocabulary at all — that's NOT the same as "genuinely
            # nothing to upgrade." It just means this model has nothing to
            # say about this specific sentence. Returning None (not []) lets
            # the caller fall through to Gemini, which isn't restricted to a
            # fixed word list and may well find something this model can't.
            return None

        suggestions = []
        for word in targets:
            alts = self._suggest_for_word(word, text)
            if alts:
                suggestions.append({
                    "original": word,
                    "alternatives": alts,
                    "tip": f"'{word}' is common and vague here — a more precise word adds impact.",
                })

        return suggestions