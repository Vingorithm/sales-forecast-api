"""
Hari libur nasional Indonesia 2025–2026.
Update setiap tahun sesuai Surat Keputusan Bersama (SKB) pemerintah.
"""

import pandas as pd

INDONESIAN_HOLIDAYS: dict = {
    # ── 2025 ──────────────────────────────────────────────────────────────────
    pd.Timestamp("2025-01-01"): "Tahun Baru Masehi",
    pd.Timestamp("2025-01-29"): "Tahun Baru Imlek 2576",
    pd.Timestamp("2025-03-29"): "Hari Raya Nyepi",
    pd.Timestamp("2025-03-31"): "Hari Raya Idul Fitri 1446 H",
    pd.Timestamp("2025-04-01"): "Hari Raya Idul Fitri 1446 H (Hari ke-2)",
    pd.Timestamp("2025-04-18"): "Wafat Isa Al Masih",
    pd.Timestamp("2025-05-01"): "Hari Buruh Internasional",
    pd.Timestamp("2025-05-12"): "Hari Raya Waisak 2569",
    pd.Timestamp("2025-05-29"): "Kenaikan Isa Al Masih",
    pd.Timestamp("2025-06-01"): "Hari Lahir Pancasila",
    pd.Timestamp("2025-06-06"): "Hari Raya Idul Adha 1446 H",
    pd.Timestamp("2025-06-27"): "Tahun Baru Islam 1447 H",
    pd.Timestamp("2025-08-17"): "Hari Kemerdekaan Republik Indonesia",
    pd.Timestamp("2025-09-05"): "Maulid Nabi Muhammad SAW",
    pd.Timestamp("2025-12-25"): "Hari Raya Natal",
    pd.Timestamp("2025-12-26"): "Cuti Bersama Natal",

    # ── 2026 ──────────────────────────────────────────────────────────────────
    pd.Timestamp("2026-01-01"): "Tahun Baru Masehi",
    pd.Timestamp("2026-01-17"): "Tahun Baru Imlek 2577",
    pd.Timestamp("2026-03-19"): "Isra Mikraj Nabi Muhammad SAW",
    pd.Timestamp("2026-03-20"): "Hari Raya Nyepi",
    pd.Timestamp("2026-04-02"): "Hari Raya Idul Fitri 1447 H",
    pd.Timestamp("2026-04-03"): "Hari Raya Idul Fitri 1447 H (Hari ke-2)",
    pd.Timestamp("2026-04-04"): "Cuti Bersama Idul Fitri",
    pd.Timestamp("2026-04-05"): "Cuti Bersama Idul Fitri",
    pd.Timestamp("2026-04-06"): "Cuti Bersama Idul Fitri",
    pd.Timestamp("2026-04-10"): "Wafat Isa Al Masih",
    pd.Timestamp("2026-05-01"): "Hari Buruh Internasional",
    pd.Timestamp("2026-05-14"): "Kenaikan Isa Al Masih",
    pd.Timestamp("2026-05-22"): "Hari Raya Waisak 2570",
    pd.Timestamp("2026-06-01"): "Hari Lahir Pancasila",
    pd.Timestamp("2026-06-10"): "Hari Raya Idul Adha 1447 H",
    pd.Timestamp("2026-06-17"): "Tahun Baru Islam 1448 H",
    pd.Timestamp("2026-08-17"): "Hari Kemerdekaan Republik Indonesia",
    pd.Timestamp("2026-09-26"): "Maulid Nabi Muhammad SAW",
    pd.Timestamp("2026-12-25"): "Hari Raya Natal",
}


def get_holiday_name(date: pd.Timestamp) -> str:
    """Return nama hari libur atau string kosong jika bukan hari libur."""
    return INDONESIAN_HOLIDAYS.get(date, "")


def is_holiday(date: pd.Timestamp) -> bool:
    return date in INDONESIAN_HOLIDAYS
