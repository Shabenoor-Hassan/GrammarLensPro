"""
ONE-TIME OFFLINE SCRIPT — run build_vocab_dictionary.py FIRST.

Uses Gemini ONLY to write example sentences for each word in
vocab_dictionary.json — it no longer judges synonyms or sophistication;
that now comes from WordNet + word frequency data, which is far more
reliable and covers many more words for the same (actually less) API use.

Requests are batched (multiple words per call) to keep this fast and
comfortably within free-tier rate limits even for hundreds of words.

Usage:
    python build_vocab_dictionary.py
    $env:GEMINI_API_KEY = "your-key"
    python generate_training_data.py

Safe to interrupt and re-run — it skips words it already has sentences for.
"""

import os
import json
import time

from google import genai

MODEL_NAME = "gemini-2.5-flash"
DICTIONARY_FILE = "vocab_dictionary.json"
OUTPUT_FILE = "vocab_training_data.jsonl"
WORDS_PER_BATCH = 10
SENTENCES_PER_WORD = 3
SLEEP_BETWEEN_CALLS_SECONDS = 4


def build_prompt(words):
    word_list = ", ".join(f'"{w}"' for w in words)
    return (
        f"For each of these words: {word_list} — write {SENTENCES_PER_WORD} short, "
        f"natural English sentences that each use that exact word. Vary the "
        f"topics (work, school, travel, daily life, technology, health).\n\n"
        f"Respond with ONLY a JSON object, no other text and no markdown fences, "
        f"mapping each word to an array of its {SENTENCES_PER_WORD} sentences:\n"
        f'{{"word1": ["sentence1", "sentence2", "sentence3"], "word2": [...]}}'
    )


def clean_json_response(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    return raw


def load_done_words(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                done.add(row.get("original", "").lower())
            except json.JSONDecodeError:
                continue
    return done


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY before running this script.")

    if not os.path.exists(DICTIONARY_FILE):
        raise SystemExit(f"{DICTIONARY_FILE} not found. Run build_vocab_dictionary.py first.")

    with open(DICTIONARY_FILE, "r", encoding="utf-8") as f:
        vocab_dictionary = json.load(f)

    all_words = list(vocab_dictionary.keys())
    done_words = load_done_words(OUTPUT_FILE)
    remaining_words = [w for w in all_words if w not in done_words]

    if done_words:
        print(f"Resuming: {len(done_words)} words already done.")
    print(f"{len(remaining_words)} words remaining out of {len(all_words)} total.\n")

    client = genai.Client(api_key=api_key)
    batches = [remaining_words[i:i + WORDS_PER_BATCH] for i in range(0, len(remaining_words), WORDS_PER_BATCH)]

    total_written = 0
    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
        for i, batch in enumerate(batches, 1):
            print(f"[batch {i}/{len(batches)}] {', '.join(batch)}")
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=build_prompt(batch),
                )
                raw = clean_json_response(response.text or "")
                sentence_map = json.loads(raw)

                batch_count = 0
                for word in batch:
                    sentences = sentence_map.get(word, [])
                    if not isinstance(sentences, list):
                        continue
                    for sentence in sentences:
                        if not isinstance(sentence, str) or word.lower() not in sentence.lower():
                            continue
                        row = {
                            "sentence": sentence.strip(),
                            "original": word,
                            "alternatives": vocab_dictionary[word],
                        }
                        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        batch_count += 1

                out_f.flush()
                total_written += batch_count
                print(f"   +{batch_count} examples (total: {total_written})")

            except Exception as e:
                print(f"   Error on this batch: {e} -- skipping.")

            time.sleep(SLEEP_BETWEEN_CALLS_SECONDS)

    print(f"\nDone. {total_written} examples written to {OUTPUT_FILE}.")
    print("Upload this file to the Colab notebook next.")


if __name__ == "__main__":
    main()