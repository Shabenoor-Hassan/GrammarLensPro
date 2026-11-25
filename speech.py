import threading
from pathlib import Path
from faster_whisper import WhisperModel

MODEL_NAME = "base.en"
_model = None
_lock = threading.Lock()

def _get_model() -> WhisperModel:
    """Lazy-load so the app starts fast and RAM is only used on first voice request."""
    global _model
    with _lock:
        if _model is None:
            _model = WhisperModel(MODEL_NAME, device="cpu",
                                  compute_type="int8", cpu_threads=4)
    return _model

def unload_model() -> None:
    """Free Whisper's RAM if memory gets tight."""
    global _model
    with _lock:
        _model = None

class TranscriptionError(Exception):
    pass

def transcribe_file(path: str | Path) -> str:
    try:
        segments, _ = _get_model().transcribe(
            str(path),
            language="en",
            beam_size=1,  # greedy decoding preserves exact acoustic speech without grammar smoothing
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt="Transcribe speech verbatim. Do not correct grammar or fix grammatical errors.",
            vad_filter=False,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
    except Exception as e:
        raise TranscriptionError(f"Could not transcribe audio: {e}") from e
    if not text:
        raise TranscriptionError("No speech detected.")
    return text