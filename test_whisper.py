import sys, time
import psutil
from faster_whisper import WhisperModel

def rss_mb():
    return psutil.Process().memory_info().rss / 1024**2

audio = sys.argv[1]            # e.g. python test_whisper.py test_audio.wav
print(f"RAM before load: {rss_mb():.0f} MB")

t0 = time.time()
model = WhisperModel("base.en", device="cpu", compute_type="int8", cpu_threads=4)
print(f"Load time: {time.time()-t0:.1f}s | RAM after load: {rss_mb():.0f} MB")

t0 = time.time()
segments, info = model.transcribe(
    audio,
    language="en",
    beam_size=5,
    temperature=0.0,                    # deterministic, less "creative" rewriting
    condition_on_previous_text=False,   # stops earlier text steering later output
    vad_filter=True,                    # skips silence, avoids hallucinated text
)
text = " ".join(s.text.strip() for s in segments)
print(f"Transcribe time: {time.time()-t0:.1f}s | RAM peak-ish: {rss_mb():.0f} MB")
print("Transcript:", text)