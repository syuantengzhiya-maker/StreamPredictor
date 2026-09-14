import logging
import os
from typing import Dict, Optional

import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spm_predictor")

# --------------------------------------------------
# App + CORS
# --------------------------------------------------
app = FastAPI(
    title="SPM Stream Predictor API",
    description="Recommends ART / SCIENCE / SEMI SCIENCE stream for a student based on Form 3 results.",
    version="5.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten this to your actual frontend domain(s) before going live
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Constants
# --------------------------------------------------
F3_COLS = ["F3_BM", "F3_BI", "F3_Math", "F3_Science", "F3_Sejarah", "F3_Geo", "F3_RBT", "F3_PSV"]
SUBJECT_LABELS = {
    "F3_BM": "Bahasa Melayu",
    "F3_BI": "English",
    "F3_Math": "Mathematics",
    "F3_Science": "Science",
    "F3_Sejarah": "Sejarah",
    "F3_Geo": "Geografi",
    "F3_RBT": "RBT",
    "F3_PSV": "PSV",
}
STREAMS = ["SCIENCE", "SEMI SCIENCE", "ART"]
MODEL_PATH = "spm_merit_predictor_by_stream.pkl"
MERIT_MIN, MERIT_MAX = 0.0, 100.0  # sane display bounds; the Ridge models can extrapolate slightly outside 0-100

# --------------------------------------------------
# AI comment settings (Groq — fast inference, generous free tier)
# Get a free key at https://console.groq.com/keys
# Set it as an environment variable on your host, e.g.:
#   GROQ_API_KEY=xxxxxxxx
# NEVER hardcode the key directly in this file.
# --------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "openai/gpt-oss-120b"  # widely available, fast; check /groq-models if this ever 404s
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_TIMEOUT_SECONDS = 15
GROQ_MAX_RETRIES = 2  # try once, then retry once more before falling back

FALLBACK_REASON = "This stream best matches your Form 3 results based on our prediction model."

# --------------------------------------------------
# Load model bundle at startup (fail loudly if missing/corrupt)
# --------------------------------------------------
# spm_merit_predictor_by_stream.pkl is a plain dict:
#   { "ART": Pipeline(StandardScaler -> Ridge),
#     "SCIENCE": Pipeline(StandardScaler -> Ridge),
#     "SEMI SCIENCE": Pipeline(StandardScaler -> Ridge) }
# Each pipeline takes the 8 F3_* columns directly and predicts that
# stream's own merit score. There is no shared feature set, no gating
# metadata, and no population stats bundled with this file.
stream_models: Dict[str, object] = {}
model_load_error: Optional[str] = None

try:
    bundle = joblib.load(MODEL_PATH)
    missing = [s for s in STREAMS if s not in bundle]
    if missing:
        raise KeyError(f"Model bundle is missing stream model(s): {missing}")
    stream_models = {s: bundle[s] for s in STREAMS}
    logger.info("Loaded per-stream model bundle from %s | streams=%s", MODEL_PATH, list(stream_models.keys()))
except Exception as e:
    model_load_error = str(e)
    logger.error("Failed to load %s: %s", MODEL_PATH, e)

if not GROQ_API_KEY:
    logger.warning(
        "GROQ_API_KEY is not set — /predict will still work, "
        "but recommendation_reason will fall back to a generic message instead of an AI comment."
    )


# --------------------------------------------------
# Request schema — validated score ranges (SPM/F3 grades are 0-100)
# --------------------------------------------------
class ScoreInput(BaseModel):
    F3_BM: float = Field(..., ge=0, le=100, description="Bahasa Melayu score (0-100)")
    F3_BI: float = Field(..., ge=0, le=100, description="Bahasa Inggeris score (0-100)")
    F3_Math: float = Field(..., ge=0, le=100, description="Mathematics score (0-100)")
    F3_Science: float = Field(..., ge=0, le=100, description="Science score (0-100)")
    F3_Sejarah: float = Field(..., ge=0, le=100, description="Sejarah score (0-100)")
    F3_Geo: float = Field(..., ge=0, le=100, description="Geography score (0-100)")
    F3_RBT: float = Field(..., ge=0, le=100, description="RBT score (0-100)")
    F3_PSV: float = Field(..., ge=0, le=100, description="PSV score (0-100)")

    class Config:
        json_schema_extra = {
            "example": {
                "F3_BM": 75, "F3_BI": 80, "F3_Math": 90, "F3_Science": 85,
                "F3_Sejarah": 70, "F3_Geo": 72, "F3_RBT": 65, "F3_PSV": 68
            }
        }


# --------------------------------------------------
# Core recommendation logic (simple: run each stream's own model,
# pick the stream with the highest predicted merit score)
# --------------------------------------------------
def predict_merit_for_stream(scores: dict, stream: str) -> float:
    x = pd.DataFrame([scores])[F3_COLS]
    pred = float(stream_models[stream].predict(x)[0])
    return round(float(np.clip(pred, MERIT_MIN, MERIT_MAX)), 2)


def recommend(scores: dict) -> dict:
    all_results = {s: predict_merit_for_stream(scores, s) for s in STREAMS}
    best_stream = max(all_results, key=all_results.get)

    return {
        "results": all_results,
        "best_stream": best_stream,
        "best_score": all_results[best_stream],
    }


# --------------------------------------------------
# AI comment generation
# --------------------------------------------------
def _score_band(mark: float) -> str:
    """Honest description of a single subject mark, used so the AI comment
    never calls a mediocre or weak score 'excellent' just because it happens
    to be the student's relative top subject."""
    if mark >= 80:
        return "excellent"
    if mark >= 65:
        return "good"
    if mark >= 50:
        return "fair / passing"
    return "weak"


def generate_ai_comment(scores: dict, best_stream: str, results: dict) -> str:
    """
    Calls Groq (Llama 3.3 / gpt-oss) to turn the recommendation into a warm,
    personalized comment explaining WHY this stream suits the student — based
    on their actual strongest subjects, not a hardcoded if/else template.

    IMPORTANT: "strongest subject" only means "highest relative to their own
    other subjects" — it does NOT mean the mark is actually good. The prompt
    below tells the model the real score band for that subject explicitly,
    so a 45/100 top subject gets described as "your comparative strength"
    or "the one you're closest to being comfortable with", never "excellent".

    Falls back to a generic sentence if the API key is missing or the call fails.
    """
    if not GROQ_API_KEY:
        return FALLBACK_REASON

    ranked_subjects = sorted(
        ((SUBJECT_LABELS[k], v) for k, v in scores.items()),
        key=lambda kv: kv[1],
        reverse=True,
    )
    top_subjects_text = ", ".join(
        f"{name} ({mark}, {_score_band(mark)})" for name, mark in ranked_subjects[:3]
    )
    overall_band = _score_band(sum(scores.values()) / len(scores))

    prompt = (
        "You are an honest, encouraging school academic advisor talking to a Form 3 student in "
        "Malaysia who is choosing their SPM stream (ART, SCIENCE, or SEMI SCIENCE).\n\n"
        f"Their Form 3 subject scores: {scores}\n"
        f"Their relatively strongest subjects, WITH the actual honest performance band for each "
        f"(this band is the true quality of the mark, not just its rank): {top_subjects_text}\n"
        f"Their overall performance band: {overall_band}\n"
        f"The recommended stream is: {best_stream}\n"
        f"Estimated merit scores per stream (for reference only): {results}\n\n"
        "Write ONE short sentence (max 30 words) explaining why this stream suits them, "
        "referencing their actual strongest subject(s) by name.\n"
        "STRICT RULES:\n"
        "- Never describe a subject as 'excellent', 'strong', or 'great' unless its band above is "
        "'excellent' or 'good'. If their best subject is only 'fair / passing' or 'weak', say "
        "something honest instead, e.g. 'the subject you're most comfortable with' or 'where you "
        "have the most room to build from' — do not inflate it.\n"
        "- If the overall band is 'weak', do not sound falsely upbeat. Be supportive and practical: "
        "acknowledge this stream is the realistic fit given where they are now, not a celebration.\n"
        "- Do not mention 'AI', 'model', or 'prediction'. Speak directly to the student as 'you'."
    )

    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        # gpt-oss-120b is a reasoning model: its internal "reasoning" tokens
        # are drawn from the SAME max_tokens budget as the final answer.
        # Keep the ceiling generous and reasoning effort low so the budget
        # goes to the actual sentence instead of deliberation.
        "max_tokens": 300,
        "reasoning_effort": "low",
        "temperature": 0.7,
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    last_error = None
    for attempt in range(1, GROQ_MAX_RETRIES + 1):
        try:
            response = requests.post(
                GROQ_URL,
                headers=headers,
                json=payload,
                timeout=GROQ_TIMEOUT_SECONDS,
            )
            if response.status_code != 200:
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                logger.warning("Groq call attempt %d failed: %s", attempt, last_error)
                continue

            data = response.json()
            try:
                comment = data["choices"][0]["message"]["content"].strip()
            except (KeyError, IndexError):
                finish_reason = data.get("choices", [{}])[0].get("finish_reason", "unknown")
                last_error = f"Unexpected response shape (finish_reason={finish_reason}): {data}"
                logger.warning("Groq call attempt %d missing content: %s", attempt, last_error)
                continue

            if comment:
                return comment
            last_error = f"Empty response body: {data}"
            logger.warning("Groq call attempt %d returned empty text: %s", attempt, data)
        except Exception as e:
            last_error = str(e)
            logger.warning("Groq call attempt %d raised exception: %s", attempt, e)

    logger.error("AI comment generation failed after %d attempts, using fallback. Last error: %s",
                 GROQ_MAX_RETRIES, last_error)
    return FALLBACK_REASON


# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.get("/")
def home():
    return {
        "message": "SPM Stream Predictor API is running",
        "model_loaded": model_load_error is None,
        "ai_comment_enabled": bool(GROQ_API_KEY),
        "streams_available": STREAMS if model_load_error is None else [],
    }


@app.get("/model-info")
def model_info():
    """Returns metadata about the loaded model bundle."""
    if model_load_error is not None:
        raise HTTPException(status_code=503, detail="Model metadata not available.")

    return {
        "approach": "one independent merit-regression model per stream; the stream with the "
                    "highest predicted merit score is recommended",
        "feature_columns": F3_COLS,
        "streams": list(stream_models.keys()),
    }


@app.get("/groq-models")
def groq_models():
    """
    Debug helper: lists the models actually available to your Groq API key right now.
    """
    if not GROQ_API_KEY:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not set.")
    try:
        resp = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "currently_configured_model": GROQ_MODEL,
            "available_models": [m["id"] for m in data.get("data", [])],
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch Groq model list: {e}")


@app.post("/predict")
def predict(data: ScoreInput):
    if model_load_error is not None:
        raise HTTPException(
            status_code=503,
            detail=f"Model is not available: {model_load_error}"
        )

    scores = data.dict()  # e.g. {"F3_BM": 75, "F3_BI": 80, ...}

    try:
        rec = recommend(scores)
    except Exception as e:
        logger.error("Recommendation failed: %s", e)
        raise HTTPException(status_code=500, detail="Prediction failed.")

    total_score = round(sum(scores.values()), 2)

    ai_reason = generate_ai_comment(scores, rec["best_stream"], rec["results"])

    logger.info(
        "Predict request | total=%s | results=%s | best=%s",
        total_score, rec["results"], rec["best_stream"],
    )

    # Response shape kept identical to the old API (results, best_stream, best_score,
    # total_input_score, recommendation_reason) so the existing frontend needs NO changes.
    return {
        "results": rec["results"],
        "best_stream": rec["best_stream"],
        "best_score": rec["best_score"],
        "total_input_score": total_score,
        "recommendation_reason": ai_reason,
    }
