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
    version="4.0.0",
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
# maps API field name -> the short column name used inside the model bundle
F3_TO_SHORT = {
    "F3_BM": "BM", "F3_BI": "BI", "F3_Math": "Math", "F3_Science": "Science",
    "F3_Sejarah": "Sejarah", "F3_Geo": "Geo", "F3_RBT": "RBT", "F3_PSV": "PSV",
}
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
MODEL_PATH = "spm_merit_predictor_v2.pkl"
SEMI_SCIENCE_BALANCE_BONUS_DEFAULT = 0.15

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
bundle: dict = {}
merit_model = None
feature_cols: list = []
f3_short_cols: list = []
gates: dict = {}
pop_mean: dict = {}
pop_std: dict = {}
merit_min: float = 0.0
merit_max: float = 100.0
semi_science_balance_bonus: float = SEMI_SCIENCE_BALANCE_BONUS_DEFAULT
model_load_error: Optional[str] = None

try:
    bundle = joblib.load(MODEL_PATH)
    merit_model = bundle["merit_model"]
    feature_cols = bundle["feature_cols"]
    f3_short_cols = bundle["f3_cols"]
    gates = bundle["gates"]
    pop_mean = bundle["population_mean"]
    pop_std = bundle["population_std"]
    merit_min, merit_max = bundle["merit_clip_range"]
    semi_science_balance_bonus = bundle.get("semi_science_balance_bonus", SEMI_SCIENCE_BALANCE_BONUS_DEFAULT)
    logger.info(
        "Loaded model bundle v2 | n_train=%s cv_mae=%.2f cv_r2=%.3f holdout_acc=%.3f",
        bundle.get("n_train"), bundle.get("cv_mae", -1), bundle.get("cv_r2", -1),
        bundle.get("holdout_recommendation_accuracy", -1),
    )
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
# Core recommendation logic (gate -> subject-fit -> merit)
# --------------------------------------------------
def eligible_streams(short_scores: dict) -> list:
    """A stream only qualifies if the student's Math+Science marks clear the
    data-driven prerequisite bar for that stream. ART has no hard prerequisite."""
    stem_total = short_scores["Math"] + short_scores["Science"]
    eligible = ["ART"]
    if stem_total >= gates["SEMI_SCIENCE_min_math_plus_science"]:
        eligible.append("SEMI SCIENCE")
    if stem_total >= gates["SCIENCE_min_math_plus_science"]:
        eligible.append("SCIENCE")
    return eligible


def subject_fit(short_scores: dict) -> tuple:
    """z-score the student's STEM marks and Humanities marks against the whole
    training population — apples-to-apples, unaffected by each stream's
    different merit baseline."""
    z = {c: (short_scores[c] - pop_mean[c]) / pop_std[c] for c in f3_short_cols}
    stem_z = float(np.mean([z["Math"], z["Science"]]))
    hum_z = float(np.mean([z["BM"], z["Sejarah"], z["Geo"]]))
    return stem_z, hum_z


def predict_merit_for_stream(short_scores: dict, stream: str) -> float:
    row = {**short_scores}
    for s in STREAMS:
        row[f"is_{s}"] = 1 if s == stream else 0
    x = pd.DataFrame([row])[feature_cols]
    pred = float(merit_model.predict(x)[0])
    return round(float(np.clip(pred, merit_min, merit_max)), 2)


def recommend(short_scores: dict) -> dict:
    eligible = eligible_streams(short_scores)
    stem_z, hum_z = subject_fit(short_scores)
    fit_score = {
        "SCIENCE": stem_z,
        "ART": hum_z,
        "SEMI SCIENCE": (stem_z + hum_z) / 2 + semi_science_balance_bonus,
    }
    best_stream = max(eligible, key=lambda s: fit_score[s])

    # numeric merit estimate for EVERY stream, for display — NOT used to pick best_stream,
    # and not directly comparable across streams (different subject mix/baseline per stream)
    all_results = {s: predict_merit_for_stream(short_scores, s) for s in STREAMS}

    return {
        "results": all_results,
        "eligible_streams": eligible,
        "best_stream": best_stream,
        "best_score": all_results[best_stream],
        "stem_fit": round(stem_z, 2),
        "humanities_fit": round(hum_z, 2),
        "forced_by_rule": len(eligible) < len(STREAMS),  # True if some stream(s) got gated out
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


def generate_ai_comment(scores: dict, best_stream: str, results: dict, forced_by_rule: bool) -> str:
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
        f"Estimated merit scores per stream (for reference only): {results}\n"
        f"{'Some streams were not eligible for this student based on their Math/Science marks.' if forced_by_rule else ''}\n\n"
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
    """Returns metadata about the trained model: CV metrics, held-out
    recommendation accuracy, and the eligibility gates in use."""
    if not bundle:
        raise HTTPException(status_code=503, detail="Model metadata not available.")

    return {
        "approach": "gate (eligibility) -> subject-fit (which stream) -> shared merit model (informational score)",
        "feature_columns": feature_cols,
        "gates": gates,
        "merit_model_cv_mae": bundle.get("cv_mae"),
        "merit_model_cv_r2": bundle.get("cv_r2"),
        "holdout_recommendation_accuracy": bundle.get("holdout_recommendation_accuracy"),
        "n_train": bundle.get("n_train"),
        "n_test": bundle.get("n_test"),
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
        short_scores = {F3_TO_SHORT[k]: v for k, v in scores.items()}
    except Exception as e:
        logger.error("Failed to map input fields: %s", e)
        raise HTTPException(status_code=400, detail="Invalid input data.")

    try:
        rec = recommend(short_scores)
    except Exception as e:
        logger.error("Recommendation failed: %s", e)
        raise HTTPException(status_code=500, detail="Prediction failed.")

    total_score = round(sum(scores.values()), 2)

    ai_reason = generate_ai_comment(scores, rec["best_stream"], rec["results"], rec["forced_by_rule"])

    logger.info(
        "Predict request | total=%s | results=%s | eligible=%s | best=%s",
        total_score, rec["results"], rec["eligible_streams"], rec["best_stream"],
    )

    # Response shape kept identical to the old API (results, best_stream, best_score,
    # total_input_score, recommendation_reason) so the existing frontend needs NO changes.
    # A few extra fields are added (eligible_streams, stem_fit, humanities_fit) — the
    # frontend can simply ignore them if it doesn't read them.
    return {
        "results": rec["results"],
        "best_stream": rec["best_stream"],
        "best_score": rec["best_score"],
        "total_input_score": total_score,
        "recommendation_reason": ai_reason,
        "eligible_streams": rec["eligible_streams"],
        "stem_fit": rec["stem_fit"],
        "humanities_fit": rec["humanities_fit"],
    }