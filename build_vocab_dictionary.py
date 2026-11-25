"""
ONE-TIME OFFLINE SCRIPT — no API, no network needed once packages are
installed. Builds a large word -> upgrade-synonym dictionary using
WordNet (a real lexical database, via nltk) and objective word-frequency
data (via wordfreq), instead of a hand-picked word list judged by an LLM.

Output: vocab_dictionary.json
    {"important": ["paramount", "consequential", "momentous"], ...}

Run this FIRST, before generate_training_data.py — it produces the word
list and synonym candidates that script will generate example sentences
for.

Usage:
    pip install nltk wordfreq
    python build_vocab_dictionary.py

Safe to re-run any time; it always rebuilds from scratch (fast — a few
seconds, no network calls after the one-time NLTK data download).
"""

import json

import nltk
from nltk.corpus import wordnet
from wordfreq import top_n_list, zipf_frequency

# One-time download of WordNet's data files, cached locally after the first run.
for resource, path in [("wordnet", "corpora/wordnet"), ("omw-1.4", "corpora/omw-1.4")]:
    try:
        nltk.data.find(path)
    except LookupError:
        print(f"Downloading NLTK resource: {resource}...")
        nltk.download(resource, quiet=True)

TOP_N_FREQUENT_WORDS = 4000     # scan this many of the most common English words
TARGET_DICTIONARY_SIZE = 800    # stop once we have this many good entries
MIN_WORD_LENGTH = 3
RARITY_MARGIN = 0.4             # a synonym must be at least this much rarer (Zipf scale) to count as an "upgrade"
MIN_SYNONYM_ZIPF = 1.5          # floor so synonyms are rare-but-real, not painfully obscure/archaic
MAX_ALTERNATIVES_PER_WORD = 3
MAX_SYNSETS_PER_WORD = 1        # ONLY the single most common sense — the tightest possible guard against wrong-sense contamination
OUTPUT_FILE = "vocab_dictionary.json"

# Confirmed-bad cases found by actually running this script and inspecting
# the output — short, multi-role words where WordNet's top-ranked sense
# doesn't match how the word is actually used day-to-day (e.g. "off"'s
# top WordNet sense is a slang verb for "kill", not the preposition/adverb
# everyone actually means). Add to this list if you spot more when you
# review vocab_dictionary.json yourself — see the note in the usage steps.
EXCLUDE_WORDS = {"off", "even", "back", "way", "going", "take", "want"}

# Safety net: WordNet includes every sense of a word, including slang and
# vulgar ones (e.g. "make" has a bathroom-slang sense, "know" has a vulgar
# biblical-euphemism sense). Restricting to the top-ranked sense above
# removes most of these, but this list is a second, explicit layer of
# defense against words that are never acceptable to suggest regardless.
BLOCKED_SYNONYMS = {
    "micturate", "piddle", "defecate", "urinate", "bonk", "hump", "shag",
    "screw", "shit", "crap", "piss", "fuck", "damn", "hell", "goddamn",
    "bloody", "bastard", "bugger", "wank", "sod",
}

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "because", "as", "of", "in", "on",
    "at", "to", "for", "with", "by", "from", "about", "into", "onto", "over", "under", "between",
    "among", "through", "during", "before", "after", "since", "until", "against", "toward",
    "towards", "within", "without", "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should", "can", "could",
    "may", "might", "must", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their", "this", "that", "these", "those", "not",
    "no", "yes", "there", "here", "what", "who", "whom", "which", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "only", "own",
    "same", "than", "too", "very", "just", "also", "one", "two", "up", "out", "get", "go", "do",
}


def get_dominant_pos(word: str):
    """Returns the POS tag ('a'=adjective, 's'=adjective satellite, 'v'=verb,
    'n'=noun, 'r'=adverb) of the word's single most common WordNet sense."""
    synsets = wordnet.synsets(word)
    return synsets[0].pos() if synsets else None


def get_wordnet_synonyms(word: str, required_pos: str):
    """Collects single-word synonyms for `word` from its dominant sense
    ONLY, and only synonyms that share the same part of speech. Restricting
    both the source word AND its synonyms to adjectives/verbs (never nouns,
    adverbs, or ambiguous function words) is what keeps genuinely wrong-sense
    results like "even" -> "eventide" or "want" -> "neediness" out — those
    happen precisely when a short, multi-role word's dominant WordNet sense
    doesn't match how it's actually used day-to-day."""
    synonyms = set()
    dominant_synsets = [s for s in wordnet.synsets(word)[:MAX_SYNSETS_PER_WORD] if s.pos() == required_pos]
    for synset in dominant_synsets:
        for lemma in synset.lemmas():
            name = lemma.name().replace("_", " ")
            if " " in name:
                continue
            name_lower = name.lower()
            if name_lower == word.lower():
                continue
            if name_lower in BLOCKED_SYNONYMS:
                continue
            # Cross-check: the synonym itself must ALSO have this same POS as
            # one of ITS dominant senses, not just appear in this synset.
            syn_pos_set = {s.pos() for s in wordnet.synsets(name_lower)[:MAX_SYNSETS_PER_WORD]}
            if required_pos not in syn_pos_set:
                continue
            synonyms.add(name_lower)
    return synonyms


def build_dictionary():
    candidates = top_n_list("en", TOP_N_FREQUENT_WORDS)
    dictionary = {}
    # Only adjectives ("good", "important") and verbs ("help", "make") are
    # reliable here. Nouns and adverbs, and especially short multi-role
    # words, are where most of the wrong-sense results came from.
    ALLOWED_POS = {"a", "s", "v"}

    for word in candidates:
        if len(dictionary) >= TARGET_DICTIONARY_SIZE:
            break

        word = word.lower()
        if word in STOPWORDS or word in EXCLUDE_WORDS or len(word) < MIN_WORD_LENGTH or not word.isalpha():
            continue

        dominant_pos = get_dominant_pos(word)
        if dominant_pos not in ALLOWED_POS:
            continue

        base_freq = zipf_frequency(word, "en")
        synonyms = get_wordnet_synonyms(word, dominant_pos)
        if not synonyms:
            continue

        # Keep only synonyms that are objectively RARER than the original —
        # that's what makes them read as an upgrade rather than a sideways
        # or simpler swap. This is a real, measurable substitute for asking
        # an LLM to judge "is this more sophisticated."
        upgrades = []
        for syn in synonyms:
            syn_freq = zipf_frequency(syn, "en")
            if syn_freq >= MIN_SYNONYM_ZIPF and (base_freq - syn_freq) >= RARITY_MARGIN:
                upgrades.append((syn, syn_freq))

        if not upgrades:
            continue

        upgrades.sort(key=lambda pair: pair[1])  # rarest (most elevated) first
        dictionary[word] = [w for w, _ in upgrades[:MAX_ALTERNATIVES_PER_WORD]]

    return dictionary


def main():
    print(f"Scanning the {TOP_N_FREQUENT_WORDS} most common English words for upgrade-worthy entries...")
    dictionary = build_dictionary()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dictionary, f, indent=2)

    print(f"\nBuilt {len(dictionary)} word entries -> {OUTPUT_FILE}")
    print("\nSample entries:")
    for word in list(dictionary.keys())[:15]:
        print(f"  {word:15} -> {dictionary[word]}")


if __name__ == "__main__":
    main()