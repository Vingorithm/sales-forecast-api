"""
Pydantic schemas untuk request & response API.
"""

from datetime import date
from typing import List, Optional
from pydantic import BaseModel, Field, validator


# ─────────────────────────────────────────────
#  REQUEST MODELS
# ─────────────────────────────────────────────

class ForecastRequest(BaseModel):
    date: str = Field(
        ...,
        description="Tanggal target format YYYY-MM-DD",
        example="2026-06-10",
    )

    @validator("date")
    def validate_date(cls, v):
        try:
            from datetime import datetime
            dt = datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Format tanggal harus YYYY-MM-DD, contoh: 2026-06-10")
        return v


class MultiForecastRequest(BaseModel):
    start_date: str = Field(
        ...,
        description="Tanggal awal forecast format YYYY-MM-DD",
        example="2026-06-01",
    )
    n_days: int = Field(
        default=7,
        ge=1,
        le=30,
        description="Jumlah hari yang diprediksi (1–30)",
    )

    @validator("start_date")
    def validate_date(cls, v):
        try:
            from datetime import datetime
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Format tanggal harus YYYY-MM-DD")
        return v


# ─────────────────────────────────────────────
#  RESPONSE MODELS
# ─────────────────────────────────────────────

class ForecastResponse(BaseModel):
    date: str = Field(..., description="Tanggal prediksi")
    predicted_revenue: float = Field(..., description="Prediksi revenue dalam Rupiah")
    lower_bound: float = Field(..., description="Batas bawah prediksi")
    upper_bound: float = Field(..., description="Batas atas prediksi")
    formatted: str = Field(..., description="Revenue dalam format Rupiah (Rp)")
    day_of_week: str = Field(..., description="Nama hari")
    is_sunday: bool = Field(..., description="Apakah hari Minggu")
    is_holiday: bool = Field(..., description="Apakah hari libur nasional")
    holiday_name: str = Field(default="", description="Nama hari libur (jika ada)")
    notes: List[str] = Field(default_factory=list, description="Catatan bisnis terkait prediksi")

    class Config:
        json_schema_extra = {
            "example": {
                "date": "2026-06-10",
                "predicted_revenue": 43250000,
                "lower_bound": 38000000,
                "upper_bound": 48500000,
                "formatted": "Rp 43.250.000",
                "day_of_week": "Wednesday",
                "is_sunday": False,
                "is_holiday": False,
                "holiday_name": "",
                "notes": [],
            }
        }


class MultiForecastResponse(BaseModel):
    start_date: str
    end_date: str
    n_days: int
    total_predicted_revenue: float
    total_formatted: str
    forecasts: List[ForecastResponse]
    summary: dict = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "start_date": "2026-06-01",
                "end_date": "2026-06-07",
                "n_days": 7,
                "total_predicted_revenue": 280000000,
                "total_formatted": "Rp 280.000.000",
                "forecasts": [],
                "summary": {
                    "avg_daily": 40000000,
                    "highest_day": "2026-06-05",
                    "lowest_day": "2026-06-01",
                    "closed_days": 1,
                },
            }
        }


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    timestamp: str


class ModelInfoResponse(BaseModel):
    model_type: str
    feature_count: int
    feature_names: List[str]
    training_window: str
    last_training_date: Optional[str]
    model_file: str
