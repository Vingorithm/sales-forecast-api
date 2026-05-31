"""
Retail POS Sales Forecasting API
=================================
FastAPI application untuk prediksi revenue harian toko retail.
Dibangun untuk deployment di Railway.

Author  : Retail AI System
Version : 1.0.0
"""

import os
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, validator

from app.forecaster import SalesForecaster
from app.schemas import (
    ForecastRequest,
    ForecastResponse,
    MultiForecastRequest,
    MultiForecastResponse,
    HealthResponse,
    ModelInfoResponse,
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("retail-forecast-api")

# ── Global forecaster instance ─────────────────────────────────────────────────
forecaster: Optional[SalesForecaster] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global forecaster
    logger.info("🚀 Starting Retail Forecast API...")
    try:
        forecaster = SalesForecaster()
        forecaster.load()
        logger.info("✅ Model loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        logger.warning("⚠️  API will start but /predict endpoints will return 503.")
    yield
    logger.info("🛑 Shutting down Retail Forecast API.")


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Retail POS Sales Forecasting API",
    description="""
## 🛒 Retail POS Sales Forecasting System

API untuk prediksi revenue harian toko retail menggunakan **XGBoost / LightGBM**
yang dilatih dengan data transaksi nyata.

### Fitur
- **Prediksi 1 hari** — input tanggal, output prediksi revenue
- **Prediksi 7 hari** — multi-step recursive forecasting
- **Business-aware** — otomatis mendeteksi hari Minggu & hari libur nasional Indonesia
- **Confidence band** — lower & upper bound prediksi

### Business Rules
- Revenue = 0 → toko **tutup** (hari libur / non-operasional)
- Hari Minggu → jam operasional lebih pendek → revenue lebih rendah
- Data training dari **Maret 2026** ke atas
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse, tags=["Root"])
async def root():
    """Landing page dengan link ke dokumentasi."""
    html = """
    <!DOCTYPE html>
    <html lang="id">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Retail Forecast API</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', system-ui, sans-serif;
                background: #0f0f1a;
                color: #e2e8f0;
                display: flex;
                align-items: center;
                justify-content: center;
                min-height: 100vh;
                padding: 2rem;
            }
            .card {
                background: #1a1a2e;
                border: 1px solid #2d2d4e;
                border-radius: 16px;
                padding: 3rem;
                max-width: 560px;
                width: 100%;
                text-align: center;
            }
            .badge {
                display: inline-block;
                background: #16213e;
                border: 1px solid #0f3460;
                color: #4fc3f7;
                font-size: 0.75rem;
                font-weight: 600;
                letter-spacing: 0.1em;
                text-transform: uppercase;
                padding: 0.3rem 0.9rem;
                border-radius: 999px;
                margin-bottom: 1.5rem;
            }
            h1 { font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem; }
            h1 span { color: #4fc3f7; }
            p { color: #94a3b8; line-height: 1.6; margin-bottom: 2rem; }
            .links { display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap; }
            a {
                display: inline-flex;
                align-items: center;
                gap: 0.4rem;
                background: #0f3460;
                color: #4fc3f7;
                text-decoration: none;
                padding: 0.65rem 1.4rem;
                border-radius: 8px;
                font-weight: 600;
                font-size: 0.9rem;
                transition: background 0.2s;
            }
            a:hover { background: #1a5276; }
            .status {
                margin-top: 2rem;
                font-size: 0.8rem;
                color: #64748b;
            }
            .dot { color: #4ade80; }
        </style>
    </head>
    <body>
        <div class="card">
            <div class="badge">🤖 AI Forecasting</div>
            <h1>Retail <span>Forecast</span> API</h1>
            <p>Prediksi revenue harian toko retail menggunakan XGBoost & LightGBM — business-aware, holiday-aware.</p>
            <div class="links">
                <a href="/docs">📖 Swagger UI</a>
                <a href="/redoc">📋 ReDoc</a>
                <a href="/health">❤️ Health</a>
                <a href="/model/info">🧠 Model Info</a>
            </div>
            <div class="status"><span class="dot">●</span> API Online — v1.0.0</div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Cek status API dan model."""
    model_loaded = forecaster is not None and forecaster.is_loaded
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    """Informasi detail tentang model yang di-load."""
    if not forecaster or not forecaster.is_loaded:
        raise HTTPException(status_code=503, detail="Model belum di-load.")
    return forecaster.get_info()


# ──────────────────────────────────────────────────────────────────────────────
#  PREDICTION ENDPOINTS
# ──────────────────────────────────────────────────────────────────────────────

@app.post(
    "/predict",
    response_model=ForecastResponse,
    tags=["Forecasting"],
    summary="Prediksi revenue 1 hari",
)
async def predict_single(request: ForecastRequest):
    """
    Prediksi revenue untuk **satu tanggal** tertentu.

    **Contoh request:**
    ```json
    { "date": "2026-06-10" }
    ```

    **Contoh response:**
    ```json
    {
      "date": "2026-06-10",
      "predicted_revenue": 43250000,
      "lower_bound": 38000000,
      "upper_bound": 48500000,
      "formatted": "Rp 43.250.000",
      "day_of_week": "Wednesday",
      "is_sunday": false,
      "is_holiday": false,
      "holiday_name": "",
      "notes": []
    }
    ```
    """
    if not forecaster or not forecaster.is_loaded:
        raise HTTPException(status_code=503, detail="Model belum di-load. Pastikan model sudah di-train dan file .pkl tersedia.")

    try:
        result = forecaster.predict_date(request.date)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get(
    "/predict",
    response_model=ForecastResponse,
    tags=["Forecasting"],
    summary="Prediksi revenue 1 hari (GET)",
)
async def predict_single_get(
    date: str = Query(..., description="Tanggal target format YYYY-MM-DD", example="2026-06-10")
):
    """Versi GET dari endpoint prediksi — cocok untuk quick test di browser."""
    return await predict_single(ForecastRequest(date=date))


@app.post(
    "/predict/multi",
    response_model=MultiForecastResponse,
    tags=["Forecasting"],
    summary="Prediksi revenue multi-hari (hingga 30 hari)",
)
async def predict_multi(request: MultiForecastRequest):
    """
    Prediksi revenue untuk **beberapa hari ke depan** secara rekursif.

    **Contoh request:**
    ```json
    {
      "start_date": "2026-06-01",
      "n_days": 7
    }
    ```
    """
    if not forecaster or not forecaster.is_loaded:
        raise HTTPException(status_code=503, detail="Model belum di-load.")

    try:
        result = forecaster.predict_range(request.start_date, request.n_days)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Multi-prediction error")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get(
    "/predict/next7",
    response_model=MultiForecastResponse,
    tags=["Forecasting"],
    summary="Prediksi 7 hari ke depan dari hari ini",
)
async def predict_next7():
    """Shortcut: prediksi otomatis 7 hari ke depan mulai besok."""
    if not forecaster or not forecaster.is_loaded:
        raise HTTPException(status_code=503, detail="Model belum di-load.")

    tomorrow = (datetime.utcnow() + timedelta(hours=7) + timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        result = forecaster.predict_range(tomorrow, 7)
        return result
    except Exception as e:
        logger.exception("next7 error")
        raise HTTPException(status_code=500, detail=str(e))
