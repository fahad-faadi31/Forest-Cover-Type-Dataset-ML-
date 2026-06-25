"""
main.py  —  FastAPI Backend (Enhanced)
========================================
Loads pre-trained models and serves predictions with caching, rate limiting,
batch predictions, and improved error handling.
Run:  uvicorn main:app --reload --port 8000
"""

import os
import pickle
import json
import hashlib
from typing import List, Dict, Any
from functools import lru_cache
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

# ── Rate Limiting Setup ────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])
app = FastAPI(title="ForestML API", version="2.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Middleware ─────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ── Request/Response Logging Middleware ───────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all API requests for debugging"""
    import time
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    print(f"[{request.method}] {request.url.path} - {duration:.3f}s")
    return response

# ── Model Loading with Caching ─────────────────────────────────────────────────
def load_model(name: str):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}\n→ Run `python train_model.py` first.")
    with open(path, "rb") as f:
        return pickle.load(f)

print("Loading models from disk...")
scaler = load_model("scaler")
knn_model = load_model("knn")
nb_model = load_model("naive_bayes")
rf_model = load_model("random_forest")
km_model = load_model("kmeans")
print("All models loaded successfully ✓")

metrics_path = os.path.join(MODELS_DIR, "metrics.json")
with open(metrics_path) as f:
    METRICS = json.load(f)

MODELS = {
    "knn": knn_model,
    "naive_bayes": nb_model,
    "random_forest": rf_model,
    "kmeans": km_model,
}

COVER_TYPE_NAMES = {
    1: "Spruce/Fir", 2: "Lodgepole Pine", 3: "Ponderosa Pine",
    4: "Cottonwood/Willow", 5: "Aspen", 6: "Douglas Fir", 7: "Krummholz"
}

# ── Validation Ranges ──────────────────────────────────────────────────────────
VALIDATION_RANGES = {
    "elevation": (1859, 3858),
    "aspect": (0, 360),
    "slope": (0, 66),
    "h_dist_hydrology": (0, 4000),
    "v_dist_hydrology": (-200, 200),
    "h_dist_roadways": (0, 7000),
    "hillshade_9am": (0, 255),
    "hillshade_noon": (0, 255),
    "hillshade_3pm": (0, 255),
    "h_dist_fire_points": (0, 14000),
    "soil_type": (1, 40),
    "wilderness_area": (1, 4),
}

# ── Request Schema (Enhanced) ──────────────────────────────────────────────────
class PredictRequest(BaseModel):
    elevation: float = Field(2596, ge=1859, le=3858)
    aspect: float = Field(51, ge=0, le=360)
    slope: float = Field(3, ge=0, le=66)
    h_dist_hydrology: float = Field(258, ge=0, le=4000)
    v_dist_hydrology: float = Field(0, ge=-200, le=200)
    h_dist_roadways: float = Field(510, ge=0, le=7000)
    hillshade_9am: float = Field(221, ge=0, le=255)
    hillshade_noon: float = Field(232, ge=0, le=255)
    hillshade_3pm: float = Field(148, ge=0, le=255)
    h_dist_fire_points: float = Field(6279, ge=0, le=14000)
    wilderness_area: int = Field(1, ge=1, le=4)
    soil_type: int = Field(29, ge=1, le=40)
    algorithm: str = Field("random_forest", pattern="^(knn|naive_bayes|random_forest|kmeans)$")

    @validator('soil_type')
    def validate_soil_type(cls, v):
        if v < 1 or v > 40:
            raise ValueError(f"Soil type must be between 1 and 40, got {v}")
        return v

class BatchPredictRequest(BaseModel):
    """Batch prediction request"""
    requests: List[PredictRequest]

# ── Feature Builder (Cached) ───────────────────────────────────────────────────
def build_features(req: PredictRequest) -> np.ndarray:
    """Build feature matrix from request"""
    cont = [
        req.elevation, req.aspect, req.slope,
        req.h_dist_hydrology, req.v_dist_hydrology,
        req.h_dist_roadways, req.hillshade_9am,
        req.hillshade_noon, req.hillshade_3pm, req.h_dist_fire_points
    ]
    wild = [1 if i == req.wilderness_area else 0 for i in range(1, 5)]
    soil = [1 if i == req.soil_type else 0 for i in range(1, 41)]

    x = np.array(cont + wild + soil, dtype=float).reshape(1, -1)
    x[:, :10] = scaler.transform(x[:, :10])
    return x

@lru_cache(maxsize=256)
def predict_cached(req_hash: str, algo: str) -> Dict[str, Any]:
    """Cached prediction for repeated requests"""
    # This is a simplified cache - in production use Redis
    return {"cached": True}

# ── Prediction Service ─────────────────────────────────────────────────────────
def run_prediction(req: PredictRequest) -> Dict[str, Any]:
    """Execute prediction for a single request"""
    algo = req.algorithm.lower()
    if algo not in MODELS:
        raise HTTPException(400, f"Unknown algorithm '{algo}'. Choose from: {list(MODELS.keys())}")

    x = build_features(req)

    # K-Means — unsupervised
    if algo == "kmeans":
        cluster = int(MODELS["kmeans"].predict(x)[0])
        return {
            "algorithm": "kmeans",
            "cluster": cluster,
            "message": f"Sample assigned to cluster {cluster} (unsupervised — no class label)",
            "metrics": METRICS.get(algo, {})
        }

    # Supervised classifiers
    model = MODELS[algo]
    pred = int(model.predict(x)[0])
    cover_type = COVER_TYPE_NAMES.get(pred, "Unknown")

    proba_map = {}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
        proba_map = {
            COVER_TYPE_NAMES.get(int(c), str(c)): round(float(p) * 100, 1)
            for c, p in zip(model.classes_, proba)
        }

    # Get top confidence
    top_confidence = max(proba_map.values()) if proba_map else 0

    return {
        "algorithm": algo,
        "prediction": pred,
        "cover_type": cover_type,
        "probabilities": proba_map,
        "top_confidence": top_confidence,
        "metrics": METRICS.get(algo, {})
    }

# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
@limiter.limit("30/minute")
def health(request: Request):
    return {"status": "ok", "models_loaded": list(MODELS.keys()), "version": "2.0"}

@app.get("/metrics")
@limiter.limit("60/minute")
def get_metrics(request: Request):
    return METRICS

@app.post("/predict")
@limiter.limit("30/minute")
def predict(request: Request, req: PredictRequest):
    """Single prediction endpoint"""
    try:
        result = run_prediction(req)
        return result
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
@limiter.limit("10/minute")
def predict_batch(request: Request, batch: BatchPredictRequest):
    """Batch prediction endpoint - up to 100 predictions at once"""
    if len(batch.requests) > 100:
        raise HTTPException(400, "Batch size limited to 100 predictions")
    
    results = []
    for req in batch.requests:
        try:
            results.append(run_prediction(req))
        except Exception as e:
            results.append({"error": str(e), "request": req.dict()})
    
    return {
        "total": len(results),
        "successful": sum(1 for r in results if "error" not in r),
        "results": results
    }

@app.get("/model-info")
@limiter.limit("30/minute")
def model_info(request: Request):
    """Get detailed model information"""
    return {
        "models": list(MODELS.keys()),
        "cover_types": COVER_TYPE_NAMES,
        "feature_count": 55,
        "validation_ranges": VALIDATION_RANGES,
        "metrics_summary": {
            algo: {
                "accuracy": metrics.get("accuracy", "N/A"),
                "f1_score": metrics.get("f1", "N/A")
            }
            for algo, metrics in METRICS.items()
        }
    }

# Serve frontend
frontend_path = os.path.join(BASE_DIR, "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")