"""
SalesForecaster — Core engine untuk prediksi revenue retail.

Memuat model joblib, membangun fitur secara dinamis,
dan menjalankan recursive multi-step forecasting.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import joblib

from app.schemas import ForecastResponse, MultiForecastResponse, ModelInfoResponse
from app.holidays import INDONESIAN_HOLIDAYS, get_holiday_name

logger = logging.getLogger("retail-forecast-api")

# ── Default paths (bisa di-override via env var) ──────────────────────────────
MODEL_PATH    = os.getenv("MODEL_PATH",    "models/sales_forecast_model.pkl")
FEATURES_PATH = os.getenv("FEATURES_PATH", "models/feature_columns.pkl")
HISTORY_PATH  = os.getenv("HISTORY_PATH",  "models/training_history.pkl")


class SalesForecaster:
    """
    Production forecaster untuk prediksi revenue retail POS.

    Usage:
        fc = SalesForecaster()
        fc.load()
        result = fc.predict_date("2026-06-10")
    """

    def __init__(self):
        self.model         = None
        self.feature_cols  = None
        self.history_df    = None
        self.is_loaded     = False
        self._model_type   = "Unknown"
        self._last_date    = None

    # ─────────────────────────────────────────────────────────────────────────
    #  Load
    # ─────────────────────────────────────────────────────────────────────────

    def load(self):
        """Load model, feature list, dan historical data dari disk."""
        logger.info(f"Loading model from  : {MODEL_PATH}")
        logger.info(f"Loading features from: {FEATURES_PATH}")

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file tidak ditemukan: {MODEL_PATH}\n"
                "Jalankan notebook training terlebih dahulu, lalu simpan model ke folder models/."
            )
        if not os.path.exists(FEATURES_PATH):
            raise FileNotFoundError(f"Feature file tidak ditemukan: {FEATURES_PATH}")

        self.model        = joblib.load(MODEL_PATH)
        self.feature_cols = joblib.load(FEATURES_PATH)
        self._model_type  = type(self.model).__name__

        # Load history jika ada (untuk lag features)
        if os.path.exists(HISTORY_PATH):
            self.history_df = joblib.load(HISTORY_PATH)
            self._last_date = self.history_df["date"].max()
            logger.info(f"History loaded: {len(self.history_df)} rows, last date: {self._last_date.date()}")
        else:
            logger.warning(
                f"History file tidak ditemukan ({HISTORY_PATH}). "
                "Lag features akan menggunakan nilai estimasi."
            )
            self.history_df = self._build_dummy_history()
            self._last_date = self.history_df["date"].max()

        self.is_loaded = True
        logger.info(f"Model loaded: {self._model_type}, features: {len(self.feature_cols)}")

    # ─────────────────────────────────────────────────────────────────────────
    #  Info
    # ─────────────────────────────────────────────────────────────────────────

    def get_info(self) -> ModelInfoResponse:
        return ModelInfoResponse(
            model_type=self._model_type,
            feature_count=len(self.feature_cols),
            feature_names=self.feature_cols,
            training_window="March 2026 – present",
            last_training_date=str(self._last_date.date()) if self._last_date else None,
            model_file=MODEL_PATH,
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  Predict single date
    # ─────────────────────────────────────────────────────────────────────────

    def predict_date(self, date_str: str) -> ForecastResponse:
        """Prediksi revenue untuk satu tanggal."""
        target_date = pd.Timestamp(date_str)
        last_date   = self.history_df["date"].max()

        # Cek jika tanggal ada di history
        hist_row = self.history_df[self.history_df["date"] == target_date]
        if len(hist_row) > 0 and hist_row["total_sales"].values[0] > 0:
            actual = float(hist_row["total_sales"].values[0])
            return self._build_response(target_date, actual, actual * 0.95, actual * 1.05, is_actual=True)

        # Hitung hari ke depan
        days_ahead = (target_date - last_date).days
        if days_ahead <= 0:
            days_ahead = 1

        forecasts = self._recursive_forecast(last_date + timedelta(days=1), days_ahead)
        target_row = next((f for f in forecasts if f["date"] == target_date), None)
        if target_row is None:
            raise ValueError(f"Gagal menghasilkan prediksi untuk {date_str}")

        return self._build_response(
            target_date,
            target_row["predicted_sales"],
            target_row["lower_bound"],
            target_row["upper_bound"],
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  Predict range
    # ─────────────────────────────────────────────────────────────────────────

    def predict_range(self, start_date_str: str, n_days: int) -> MultiForecastResponse:
        """Prediksi revenue untuk beberapa hari secara rekursif."""
        start_date = pd.Timestamp(start_date_str)
        last_date  = self.history_df["date"].max()

        # Mulai forecast dari hari setelah last_date (atau start_date, mana yang lebih dulu)
        forecast_start = min(start_date, last_date + timedelta(days=1))
        days_needed    = (start_date + timedelta(days=n_days - 1) - last_date).days

        raw_forecasts = self._recursive_forecast(forecast_start, max(days_needed, n_days))

        # Filter ke window yang diminta
        end_date = start_date + timedelta(days=n_days - 1)
        filtered = [f for f in raw_forecasts
                    if start_date <= f["date"] <= end_date]

        responses = [
            self._build_response(f["date"], f["predicted_sales"], f["lower_bound"], f["upper_bound"])
            for f in filtered
        ]

        total = sum(r.predicted_revenue for r in responses)
        dates_open = [r for r in responses if not r.is_holiday]
        avg_daily  = total / len(dates_open) if dates_open else 0
        highest    = max(responses, key=lambda r: r.predicted_revenue) if responses else None
        lowest     = min(
            [r for r in responses if not r.is_holiday],
            key=lambda r: r.predicted_revenue,
            default=None
        )

        return MultiForecastResponse(
            start_date=start_date_str,
            end_date=str(end_date.date()),
            n_days=len(responses),
            total_predicted_revenue=total,
            total_formatted=self._fmt(total),
            forecasts=responses,
            summary={
                "avg_daily"          : round(avg_daily),
                "avg_daily_formatted": self._fmt(avg_daily),
                "highest_day"        : str(highest.date) if highest else None,
                "highest_revenue"    : self._fmt(highest.predicted_revenue) if highest else None,
                "lowest_day"         : str(lowest.date) if lowest else None,
                "lowest_revenue"     : self._fmt(lowest.predicted_revenue) if lowest else None,
                "closed_days"        : sum(1 for r in responses if r.is_holiday),
                "sunday_days"        : sum(1 for r in responses if r.is_sunday),
            },
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  Recursive forecast engine
    # ─────────────────────────────────────────────────────────────────────────

    def _recursive_forecast(self, start_date: pd.Timestamp, n_days: int) -> list:
        """
        Multi-step recursive forecasting.
        Setiap prediksi digunakan sebagai lag untuk hari berikutnya.
        """
        df = self.history_df.copy()
        forecasts = []

        for i in range(n_days):
            target_date = start_date + timedelta(days=i)
            features    = self._build_feature_row(df, target_date)

            X = pd.DataFrame([features])[self.feature_cols]
            pred = float(self.model.predict(X)[0])
            pred = max(pred, 0)

            # Confidence band: ±0.5 std dari 14 hari terakhir
            recent_std = df["total_sales"].iloc[-14:].std()
            lower = max(pred - recent_std * 0.5, 0)
            upper = pred + recent_std * 0.5

            forecasts.append({
                "date"           : target_date,
                "predicted_sales": pred,
                "lower_bound"    : lower,
                "upper_bound"    : upper,
            })

            # Append ke history agar lag berikutnya akurat
            new_row = {"date": target_date, "total_sales": pred, **features}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        return forecasts

    # ─────────────────────────────────────────────────────────────────────────
    #  Feature builder
    # ─────────────────────────────────────────────────────────────────────────

    def _build_feature_row(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[str, Any]:
        """Build feature dict untuk satu tanggal berdasarkan history saat ini."""
        sales = df["total_sales"]
        dow   = target_date.dayofweek

        is_sunday  = int(dow == 6)
        is_sat     = int(dow == 5)
        is_weekend = int(dow >= 5)
        is_holiday = int(target_date in INDONESIAN_HOLIDAYS)

        def safe_lag(n):
            return float(sales.iloc[-n]) if len(sales) >= n else 0.0

        def safe_roll_mean(n):
            return float(sales.iloc[-n:].mean()) if len(sales) >= 1 else 0.0

        def safe_roll_std(n):
            return float(sales.iloc[-n:].std()) if len(sales) >= 2 else 0.0

        features = {
            # Lag
            "lag_1"              : safe_lag(1),
            "lag_2"              : safe_lag(2),
            "lag_3"              : safe_lag(3),
            "lag_7"              : safe_lag(7),
            "lag_14"             : safe_lag(14),
            "lag_30"             : safe_lag(30),
            # Rolling
            "rolling_mean_7"     : safe_roll_mean(7),
            "rolling_mean_14"    : safe_roll_mean(14),
            "rolling_std_7"      : safe_roll_std(7),
            "rolling_max_7"      : float(sales.iloc[-7:].max()) if len(sales) >= 1 else 0.0,
            "rolling_min_7"      : float(sales.iloc[-7:].min()) if len(sales) >= 1 else 0.0,
            # Calendar
            "day_of_week"        : dow,
            "day_of_month"       : target_date.day,
            "week_of_month"      : (target_date.day - 1) // 7 + 1,
            "month"              : target_date.month,
            "quarter"            : (target_date.month - 1) // 3 + 1,
            "is_weekend"         : is_weekend,
            "is_sunday"          : is_sunday,
            "is_saturday"        : is_sat,
            "day_sin"            : np.sin(2 * np.pi * dow / 7),
            "day_cos"            : np.cos(2 * np.pi * dow / 7),
            "month_sin"          : np.sin(2 * np.pi * target_date.month / 12),
            "month_cos"          : np.cos(2 * np.pi * target_date.month / 12),
            # Business
            "is_store_closed"    : is_holiday,
            "sunday_short_ops"   : int(is_sunday and not is_holiday),
            "national_holiday"   : is_holiday,
            "pre_holiday"        : int((target_date + timedelta(1)) in INDONESIAN_HOLIDAYS),
            "post_holiday"       : int((target_date - timedelta(1)) in INDONESIAN_HOLIDAYS),
            "revenue_growth_rate": self._growth_rate(sales),
            "moving_avg_7"       : safe_roll_mean(7),
            "moving_avg_14"      : safe_roll_mean(14),
            "weekly_sales_trend" : self._trend(sales, 7),
            "days_since_closure" : self._days_since_closure(df),
            "same_day_last_week" : safe_lag(7),
        }
        return features

    # ─────────────────────────────────────────────────────────────────────────
    #  Helper calculations
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _growth_rate(sales: pd.Series) -> float:
        if len(sales) < 2 or sales.iloc[-2] == 0:
            return 0.0
        rate = (sales.iloc[-1] - sales.iloc[-2]) / abs(sales.iloc[-2])
        return float(np.clip(rate, -3, 3))

    @staticmethod
    def _trend(sales: pd.Series, n: int) -> float:
        window = sales.iloc[-n:].values
        if len(window) < 2:
            return 0.0
        t = np.arange(len(window))
        return float(np.polyfit(t, window, 1)[0])

    @staticmethod
    def _days_since_closure(df: pd.DataFrame) -> int:
        if "is_store_closed" not in df.columns:
            return 1
        closed_series = (df["is_store_closed"] == 1).values[::-1]
        for i, c in enumerate(closed_series):
            if c:
                return i
        return len(df)

    # ─────────────────────────────────────────────────────────────────────────
    #  Response builder
    # ─────────────────────────────────────────────────────────────────────────

    def _build_response(
        self,
        target_date: pd.Timestamp,
        pred: float,
        lower: float,
        upper: float,
        is_actual: bool = False,
    ) -> ForecastResponse:
        dow_names   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow_name    = dow_names[target_date.dayofweek]
        is_sunday   = target_date.dayofweek == 6
        is_holiday  = target_date in INDONESIAN_HOLIDAYS
        hol_name    = get_holiday_name(target_date)

        notes = []
        if is_actual:
            notes.append("Data aktual dari history transaksi.")
        if is_holiday:
            notes.append(f"Hari libur nasional: {hol_name}. Toko kemungkinan tutup.")
        elif pred == 0:
            notes.append("Prediksi revenue = 0, kemungkinan toko tutup.")
        if is_sunday and not is_holiday:
            notes.append("Hari Minggu — jam operasional lebih pendek, revenue lebih rendah dari biasa.")
        if target_date.dayofweek == 4:  # Jumat
            notes.append("Hari Jumat — potensi lonjakan pembelian sore/malam.")

        return ForecastResponse(
            date=str(target_date.date()),
            predicted_revenue=round(pred),
            lower_bound=round(lower),
            upper_bound=round(upper),
            formatted=self._fmt(pred),
            day_of_week=dow_name,
            is_sunday=is_sunday,
            is_holiday=is_holiday,
            holiday_name=hol_name,
            notes=notes,
        )

    @staticmethod
    def _fmt(value: float) -> str:
        """Format angka ke Rupiah dengan pemisah titik."""
        return f"Rp {int(value):,}".replace(",", ".")

    # ─────────────────────────────────────────────────────────────────────────
    #  Dummy history (fallback jika training_history.pkl tidak ada)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_dummy_history() -> pd.DataFrame:
        """
        Buat dummy history 60 hari untuk keperluan lag features.
        Digunakan ketika training_history.pkl tidak tersedia.
        Idealnya ganti dengan data history nyata.
        """
        end_date   = pd.Timestamp.today().normalize() - timedelta(days=1)
        start_date = end_date - timedelta(days=59)
        dates      = pd.date_range(start_date, end_date)

        np.random.seed(42)
        sales = []
        base  = 35_000_000
        for d in dates:
            if d.dayofweek == 6:           # Minggu
                v = base * np.random.uniform(0.55, 0.75)
            elif d.dayofweek >= 5:          # Sabtu
                v = base * np.random.uniform(1.1, 1.4)
            else:
                v = base * np.random.uniform(0.8, 1.2)
            sales.append(v)

        df = pd.DataFrame({"date": dates, "total_sales": sales})
        df["is_store_closed"] = 0
        df["day_of_week"]     = df["date"].dt.dayofweek
        df["is_sunday"]       = (df["day_of_week"] == 6).astype(int)
        df["days_since_closure"] = range(len(df))
        return df
