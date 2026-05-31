# 🛒 Retail POS Sales Forecasting API

FastAPI production service untuk prediksi revenue harian toko retail,
menggunakan model **XGBoost / LightGBM** hasil training di Colab.

---

## 📁 Struktur Project

```
retail-forecast-api/
├── main.py                  ← FastAPI app (entry point)
├── app/
│   ├── forecaster.py        ← Core prediction engine
│   ├── schemas.py           ← Pydantic request/response models
│   └── holidays.py          ← Registry hari libur nasional Indonesia
├── models/                  ← Folder untuk file .pkl (TIDAK di-commit)
│   ├── sales_forecast_model.pkl
│   ├── feature_columns.pkl
│   └── training_history.pkl
├── export_model.py          ← Script export model dari Colab
├── requirements.txt
├── Procfile
└── railway.json
```

---

## 🚀 Cara Deploy ke Railway

### Step 1 — Export Model dari Colab

Setelah notebook training selesai, jalankan di Colab:

```python
# Salin isi export_model.py ke cell baru di Colab, lalu run
# Kemudian download folder models/
from google.colab import files
import shutil
shutil.make_archive("models", "zip", "models")
files.download("models.zip")
```

### Step 2 — Siapkan Repository

```bash
git init
git add .
git commit -m "Initial commit: Retail Forecast API"
```

### Step 3 — Deploy ke Railway

1. Buka [railway.app](https://railway.app) → **New Project**
2. Pilih **Deploy from GitHub repo** → pilih repo ini
3. Railway otomatis detect `Procfile` dan mulai build

### Step 4 — Upload Model Files

Karena file `.pkl` tidak di-commit ke git, gunakan salah satu cara:

**Opsi A — Railway Volume (recommended):**
```
Railway Dashboard → Project → Add Volume → Mount Path: /app/models
Lalu upload file .pkl via Railway CLI:
railway run -- cp models/*.pkl /app/models/
```

**Opsi B — Environment variable path:**
```
Set di Railway dashboard:
MODEL_PATH    = /tmp/sales_forecast_model.pkl
FEATURES_PATH = /tmp/feature_columns.pkl
HISTORY_PATH  = /tmp/training_history.pkl
```

**Opsi C — Include dalam repo (untuk demo/testing):**
```
# Hapus *.pkl dari .gitignore, lalu commit file pkl
# Tidak disarankan untuk production (ukuran repo besar)
```

---

## 🌐 Endpoint API

| Method | Path            | Deskripsi                          |
|--------|-----------------|------------------------------------|
| GET    | `/`             | Landing page                       |
| GET    | `/health`       | Status API & model                 |
| GET    | `/model/info`   | Info model (type, features, dll)   |
| POST   | `/predict`      | Prediksi 1 hari                    |
| GET    | `/predict?date=`| Prediksi 1 hari (GET shortcut)     |
| POST   | `/predict/multi`| Prediksi multi-hari (max 30)       |
| GET    | `/predict/next7`| Prediksi 7 hari ke depan otomatis  |
| GET    | `/docs`         | Swagger UI                         |
| GET    | `/redoc`        | ReDoc dokumentasi                  |

---

## 📖 Contoh Request & Response

### Prediksi 1 hari

```bash
curl -X POST https://your-app.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{"date": "2026-06-10"}'
```

```json
{
  "date": "2026-06-10",
  "predicted_revenue": 43250000,
  "lower_bound": 38100000,
  "upper_bound": 48400000,
  "formatted": "Rp 43.250.000",
  "day_of_week": "Wednesday",
  "is_sunday": false,
  "is_holiday": true,
  "holiday_name": "Hari Raya Idul Adha 1447 H",
  "notes": ["Hari libur nasional: Hari Raya Idul Adha 1447 H. Toko kemungkinan tutup."]
}
```

### Prediksi 7 hari

```bash
curl -X POST https://your-app.railway.app/predict/multi \
  -H "Content-Type: application/json" \
  -d '{"start_date": "2026-06-01", "n_days": 7}'
```

---

## ⚙️ Environment Variables

| Variable       | Default                              | Keterangan                  |
|----------------|--------------------------------------|-----------------------------|
| `PORT`         | 8000                                 | Otomatis diset Railway      |
| `MODEL_PATH`   | `models/sales_forecast_model.pkl`    | Path ke file model          |
| `FEATURES_PATH`| `models/feature_columns.pkl`         | Path ke feature list        |
| `HISTORY_PATH` | `models/training_history.pkl`        | Path ke training history    |

---

## 🧪 Test Lokal

```bash
pip install -r requirements.txt

# Jalankan API
uvicorn main:app --reload --port 8000

# Test di browser
open http://localhost:8000/docs
```

---

## 📝 Business Rules

- **Revenue = 0** → toko tutup (hari libur nasional / non-operasional) — bukan anomali
- **Hari Minggu** → jam operasional lebih pendek → revenue lebih rendah dari normal
- **Data training** → hanya data mulai Maret 2026 (POS integration reliable)
- **Evaluasi** → menggunakan SMAPE (aman untuk revenue = 0)
