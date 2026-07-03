"""
CrowdSight FastAPI application.

Endpoints
---------
GET  /health    — liveness + model status
POST /analyze   — upload an image, get crowd count + density heatmap

Run locally:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Interactive docs: http://localhost:8000/docs
"""

import io
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Security, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image

from api.inference import CrowdInferenceEngine

# ─── Configuration ────────────────────────────────────────────────────────────

WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", "checkpoints/best.pth")
API_KEY      = os.getenv("API_KEY", "")          # leave blank to disable auth
MAX_IMAGE_MB = int(os.getenv("MAX_IMAGE_MB", "20"))

_engine: Optional[CrowdInferenceEngine] = None

# ─── Lifespan (startup / shutdown) ───────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    _engine = CrowdInferenceEngine(weights_path=WEIGHTS_PATH)
    loaded = "✓ weights loaded" if _engine.weights_loaded else "⚠ random weights (no checkpoint)"
    print(f"[CrowdSight] device={_engine.device_name}  {loaded}")
    yield
    _engine = None

# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CrowdSight API",
    description=(
        "Real-time crowd density estimation via Mixed Attention-Based Multi-Column CNN.\n\n"
        "Based on: *Improving Crowd Counting Efficiency Using Spatial Attention-Based "
        "Multi-Column CNN*, R. Khalimjanov, J. Gwak, M. Jeon — KINGPC 2024."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Auth ─────────────────────────────────────────────────────────────────────

security = HTTPBearer(auto_error=False)

def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Security(security)):
    if not API_KEY:
        return   # auth disabled
    if credentials is None or credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    """Liveness probe — returns model and device status."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {
        "status": "ok",
        "device": _engine.device_name,
        "weights_loaded": _engine.weights_loaded,
        "version": app.version,
    }


@app.post("/analyze", tags=["Crowd Analysis"])
async def analyze(
    image: UploadFile = File(..., description="JPEG or PNG image to analyse"),
    _: None = Security(verify_token),
):
    """
    Estimate crowd count from an uploaded image.

    **Returns**
    - `count` — estimated number of people (float)
    - `density_map` — base64-encoded PNG heatmap (jet colormap)
    - `inference_time_ms` — wall-clock inference time
    """
    # Validate content type
    if image.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported media type '{image.content_type}'. Use JPEG or PNG.",
        )

    raw = await image.read()

    # Size guard
    if len(raw) > MAX_IMAGE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds {MAX_IMAGE_MB} MB limit.",
        )

    try:
        pil_img = Image.open(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not decode image: {exc}",
        )

    if _engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not ready.")

    result = _engine.analyze(pil_img)
    return result
