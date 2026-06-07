#!/usr/bin/env python3
"""
Model Export Script
Eğitilmiş modelleri lab/monitor/models/ klasörüne export eder.
Çalıştırma: python model_export.py
"""

import os
import json
import pickle
import shutil
import numpy as np
from pathlib import Path

EXPORT_DIR = Path("lab/monitor/models")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

exported = []
skipped  = []

# ── LightGBM (UNSW) ──────────────────────────────────────────────────────────
try:
    import lightgbm as lgb
    src = Path("models/lightgbm_model_unsw.txt")
    if src.exists():
        shutil.copy(src, EXPORT_DIR / "lgbm_unsw.txt")
        exported.append("LightGBM (UNSW)")
    else:
        skipped.append("LightGBM (UNSW) — models/lightgbm_model_unsw.txt bulunamadı")
except ImportError:
    skipped.append("LightGBM — kütüphane yok")

# ── CNN (UNSW) ────────────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    for candidate in ["models/cnn_model_unsw.keras", "models/cnn_model_unsw.h5"]:
        p = Path(candidate)
        if p.exists():
            shutil.copy(p, EXPORT_DIR / ("cnn_unsw" + p.suffix))
            exported.append(f"CNN (UNSW) — {p.name}")
            break
    else:
        skipped.append("CNN (UNSW) — .keras/.h5 bulunamadı")
except ImportError:
    skipped.append("CNN — TensorFlow yok")

# ── LSTM (UNSW) ───────────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    for candidate in ["models/lstm_model_unsw.keras", "models/lstm_model_unsw.h5"]:
        p = Path(candidate)
        if p.exists():
            shutil.copy(p, EXPORT_DIR / ("lstm_unsw" + p.suffix))
            exported.append(f"LSTM (UNSW) — {p.name}")
            break
    else:
        skipped.append("LSTM (UNSW) — .keras/.h5 bulunamadı")
except ImportError:
    pass  # zaten CNN için yazdı

# ── Scaler ────────────────────────────────────────────────────────────────────
for candidate in ["processed_data_unsw/scaler.pkl", "processed_data_unsw/scaler.joblib"]:
    p = Path(candidate)
    if p.exists():
        shutil.copy(p, EXPORT_DIR / "scaler_unsw.pkl")
        exported.append("Scaler (UNSW)")
        break
else:
    skipped.append("Scaler — processed_data_unsw/scaler.pkl bulunamadı")

# ── Feature names ─────────────────────────────────────────────────────────────
feat_src = Path("processed_data_unsw/feature_names.txt")
if feat_src.exists():
    shutil.copy(feat_src, EXPORT_DIR / "feature_names_unsw.txt")
    exported.append("Feature names (UNSW)")
else:
    # Fallback: live extractor'ın ürettiği sütunları yaz
    live_features = [
        "flow_duration","flow_bytes_s","flow_pkts_s",
        "fwd_pkt_count","fwd_len_mean","fwd_len_std","fwd_len_min","fwd_len_max","fwd_bytes",
        "fwd_iat_mean","fwd_iat_std",
        "bwd_pkt_count","bwd_len_mean","bwd_len_std","bwd_len_min","bwd_len_max","bwd_bytes",
        "bwd_iat_mean","bwd_iat_std",
        "pkt_len_mean","pkt_len_std","pkt_len_min","pkt_len_max",
        "iat_mean","iat_std",
        "syn_cnt","fin_cnt","rst_cnt","psh_cnt","ack_cnt","urg_cnt",
        "down_up_ratio","total_pkts","total_bytes",
    ]
    with open(EXPORT_DIR / "feature_names_unsw.txt", "w") as f:
        f.write("\n".join(live_features))
    skipped.append("Feature names — fallback (live extractor sütunları) kullanıldı")

# ── Özet ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  MODEL EXPORT SONUCU")
print("=" * 60)
print(f"\n✅ Export edildi ({len(exported)}):")
for e in exported:
    print(f"   • {e}")
if skipped:
    print(f"\n⚠️  Atlandı ({len(skipped)}):")
    for s in skipped:
        print(f"   • {s}")
print(f"\n📁 Hedef klasör: {EXPORT_DIR.resolve()}")
print("=" * 60)

# manifest.json
manifest = {
    "exported": exported,
    "skipped": skipped,
    "export_dir": str(EXPORT_DIR.resolve()),
}
with open(EXPORT_DIR / "manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
