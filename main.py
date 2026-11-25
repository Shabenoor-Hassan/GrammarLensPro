from fastapi import FastAPI, Depends, HTTPException, Header, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import Optional
import os
import tempfile
import database
import auth
from engine import GrammarEngine, ExplanationGenerator, explain_error
import insights
from speech import transcribe_file, TranscriptionError

app = FastAPI(title="GrammarLens AI Executive")

database.init_db()
engine = GrammarEngine()  # 1 Local Seq2Seq model (vennify/t5-base-grammar-correction)
# Pedagogical explanation generator powered by Gemini API on-demand.
# Gracefully falls back to rule-based templates if offline or API key is absent.
explainer = ExplanationGenerator()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_no_cache_header(request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.endswith((".html", ".js", ".css")):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10 MB

@app.post("/api/speech-to-text")
async def speech_to_text(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty audio file.")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "Audio too large (max 10 MB).")

    ctype = (file.content_type or "").lower()
    fname = (file.filename or "").lower()
    
    if "wav" in ctype or fname.endswith(".wav"):
        suffix = ".wav"
    elif "ogg" in ctype or fname.endswith(".ogg"):
        suffix = ".ogg"
    elif "mp4" in ctype or fname.endswith(".mp4") or fname.endswith(".m4a"):
        suffix = ".mp4"
    elif "mp3" in ctype or fname.endswith(".mp3"):
        suffix = ".mp3"
    else:
        suffix = ".webm"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(data)
        tmp.close()
        text = await run_in_threadpool(transcribe_file, tmp.name)  # keeps server responsive
    except TranscriptionError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(500, f"Speech processing error: {e}")
    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass

    return {"transcript": text}

class UserRegister(BaseModel):
    username: str
    email: str
    password: str
    english_level: Optional[str] = "intermediate"

class UserLogin(BaseModel):
    username: str
    password: str

class GrammarCheckRequest(BaseModel):
    text: str
    user_level: Optional[str] = "intermediate"

class ExplainNaturalRequest(BaseModel):
    original: str
    replacement: str
    error_type: str
    sentence: str
    user_level: Optional[str] = "intermediate"

class ProfileUpdate(BaseModel):
    english_level: str

def get_current_user(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ")[1]
    return auth.decode_access_token(token)

@app.post("/api/register")
def register(data: UserRegister):
    username = data.username.strip()
    email = data.email.strip().lower()
    password = data.password.strip()

    conn = database.get_db()
    cursor = conn.cursor()

    # Case-insensitive duplicate check
    cursor.execute(
        "SELECT id FROM users WHERE LOWER(username) = ? OR LOWER(email) = ?", 
        (username.lower(), email)
    )
    if cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="Username or email already registered")

    hashed_pw = auth.get_password_hash(password)
    cursor.execute(
        "INSERT INTO users (username, email, hashed_password, english_level) VALUES (?, ?, ?, ?)",
        (username, email, hashed_pw, data.english_level or "intermediate")
    )
    conn.commit()
    user_id = cursor.lastrowid
    conn.close()

    token = auth.create_access_token({"sub": username, "user_id": user_id, "english_level": data.english_level})
    return {"message": "Success", "access_token": token, "username": username, "english_level": data.english_level}

@app.post("/api/login")
def login(data: UserLogin):
    # Allows logging in with EITHER Username OR Email (case-insensitive)
    identifier = data.username.strip().lower()
    password = data.password.strip()

    conn = database.get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, username, hashed_password, english_level FROM users WHERE LOWER(username) = ? OR LOWER(email) = ?", 
        (identifier, identifier)
    )
    user = cursor.fetchone()
    conn.close()

    if not user or not auth.verify_password(password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = auth.create_access_token({"sub": user["username"], "user_id": user["id"], "english_level": user["english_level"]})
    return {"access_token": token, "username": user["username"], "english_level": user["english_level"] or "intermediate"}

@app.post("/api/check-grammar")
def check_grammar(req: GrammarCheckRequest, user=Depends(get_current_user)):
    text = req.text.strip()
    if not text:
        return {
            "original": "", "corrected": "", "errors": [], "total_errors": 0,
            "metrics": {}, "vocab_upgrades": []
        }

    level = (req.user_level or "intermediate").lower()

    # 1. Run Transformer Engine
    corrected = engine.correct_sentence(text)

    # 2. Detect errors and classify them into specific grammar categories.
    #    Tips here are rule-based (fast, no model calls) so the main Analyze
    #    click always stays quick no matter how many mistakes are found.
    #    Natural AI phrasing is available on demand per-mistake via
    #    /api/explain-natural instead of being generated for every error here.
    detected_errors = engine.detect_errors(text, corrected, level)

    # 3. Calculate Readability & Vocabulary Insights
    metrics = insights.calculate_text_metrics(text)
    vocab_upgrades = insights.find_vocabulary_upgrades(text, level)

    # 4. Save to history if logged in
    if user:
        conn = database.get_db()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO history (user_id, original_text, corrected_text, error_count, cefr_score) VALUES (?, ?, ?, ?, ?)",
            (user["user_id"], text, corrected, len(detected_errors), metrics.get("cefr_level"))
        )
        conn.commit()
        conn.close()

    return {
        "original": text,
        "corrected": corrected,
        "errors": detected_errors,
        "total_errors": len(detected_errors),
        "metrics": metrics,
        "vocab_upgrades": vocab_upgrades
    }

@app.post("/api/explain-natural")
def explain_natural(req: ExplainNaturalRequest):
    """On-demand natural-language explanation for a single mistake. Only
    fires one generation call per request (never a batch), which keeps
    latency low and predictable regardless of how many mistakes a sentence
    has — that's the whole point of making this its own endpoint instead of
    generating a tip for every error during the main analysis."""
    level = (req.user_level or "intermediate").lower()

    natural_tip = explainer.generate(req.error_type, req.original, req.replacement, req.sentence, level)
    if natural_tip:
        return {"pedagogical_tip": natural_tip, "tip_source": "ai"}

    # Generation unavailable or produced something unusable this time —
    # fall back to the same rule-based template used in check-grammar.
    _, fallback_tip = explain_error(req.error_type, req.original, req.replacement, level)
    return {"pedagogical_tip": fallback_tip, "tip_source": "rule"}

@app.get("/api/history")
def get_history(user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    conn = database.get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, original_text, corrected_text, error_count, cefr_score, created_at FROM history WHERE user_id = ? ORDER BY id DESC LIMIT 8",
        (user["user_id"],)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "static"), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)