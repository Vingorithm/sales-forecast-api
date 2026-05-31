"""
export_model.py
===============
Jalankan script ini di Colab SETELAH notebook training selesai,
untuk mengekspor model ke format yang siap di-deploy di Railway.

Usage (di Colab):
    !python export_model.py

Output folder `models/` berisi:
    - sales_forecast_model.pkl   ← trained model (XGBoost / LightGBM)
    - feature_columns.pkl        ← list nama fitur
    - training_history.pkl       ← daily history DataFrame (untuk lag features)
"""

import os
import joblib
import pandas as pd

# ── Pastikan folder models/ ada ────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

# ── Variabel ini sudah ada setelah notebook training dijalankan ───────────────
# best_model   : model terbaik (XGBoost / LightGBM / RF)
# FEATURE_COLS : list nama fitur
# daily        : DataFrame daily revenue (sudah dengan semua fitur)

print("Exporting model artifacts...")

# 1. Model
joblib.dump(best_model, "models/sales_forecast_model.pkl")
print("✅ models/sales_forecast_model.pkl")

# 2. Feature columns
joblib.dump(FEATURE_COLS, "models/feature_columns.pkl")
print("✅ models/feature_columns.pkl")

# 3. Training history (hanya kolom yang dibutuhkan)
history_cols = ["date", "total_sales", "is_store_closed", "day_of_week",
                "is_sunday", "days_since_closure", "national_holiday"]
history_export = daily[[c for c in history_cols if c in daily.columns]].copy()
joblib.dump(history_export, "models/training_history.pkl")
print(f"✅ models/training_history.pkl  ({len(history_export)} rows)")

print()
print("="*50)
print("Export selesai! Download folder models/ lalu")
print("letakkan di root project FastAPI Anda.")
print("="*50)
