#!/usr/bin/env python3
"""
UNSW-NB15 Dataset Replay — Traffic Simulator
Test CSV'yi sanki canlı trafik geliyormus gibi modele besler.
Her batch 5-15 kayit icerir, gercekci zamanlamalarla akar.
"""

import json
import logging
import os
import random
import signal
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from predictor import Predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/app/logs/simulator.log"),
    ],
)
log = logging.getLogger("SIM")

# ── Ayarlar ───────────────────────────────────────────────────────────────────
CSV_PATH    = Path(os.environ.get("DATASET_CSV", "/app/data/UNSW_NB15_testing-set.csv"))
ALERTS_FILE = Path(os.environ.get("ALERTS_FILE", "/app/logs/alerts.jsonl"))
IDS_MODE    = os.environ.get("IDS_MODE", "ensemble")
BATCH_SIZE  = int(os.environ.get("BATCH_SIZE", "12"))   # Her seferinde kaç kayıt
BATCH_DELAY = float(os.environ.get("BATCH_DELAY", "4")) # Saniye cinsinden bekleme
MODE_FILE   = ALERTS_FILE.parent / "ids_mode.txt"       # Dashboard'dan mod seçimi


def get_current_mode(predictor) -> str:
    """Dashboard'dan mod değişikliği var mı kontrol et."""
    try:
        if MODE_FILE.exists():
            mode = MODE_FILE.read_text().strip()
            if mode in ("ensemble", "fast", "accurate") and mode != predictor.mode:
                predictor.switch_mode(mode)
                log.info(f"Mod guncellendi: {mode}")
            return mode
    except Exception:
        pass
    return predictor.mode

# Gerçekçi IP havuzu
INTERNAL_IPS = [
    "172.20.0.10", "172.20.0.20", "172.20.0.30",
    "192.168.1.10", "192.168.1.20", "10.0.0.5",
]
EXTERNAL_IPS = [
    "45.33.32.156", "104.21.7.89", "185.220.101.34",
    "23.227.38.65",  "198.51.100.1", "203.0.113.42",
]

_stop = threading.Event()
signal.signal(signal.SIGINT,  lambda s, f: _stop.set())
signal.signal(signal.SIGTERM, lambda s, f: _stop.set())


# ── Veri yükleme ──────────────────────────────────────────────────────────────

def load_dataset() -> pd.DataFrame:
    log.info(f"Dataset yukleniyor: {CSV_PATH}")
    if not CSV_PATH.exists():
        log.error(f"CSV bulunamadi: {CSV_PATH}")
        return pd.DataFrame()

    df = pd.read_csv(CSV_PATH, low_memory=False)
    log.info(f"Dataset yuklendi: {len(df):,} kayit, {len(df.columns)} sutun")

    # Label sütununu bul
    label_col = None
    for c in ["label", "Label", "attack_cat", "class"]:
        if c in df.columns:
            label_col = c
            break

    normal_count  = (df[label_col] == 0).sum() if label_col else 0
    attack_count  = (df[label_col] == 1).sum() if label_col else 0
    log.info(f"Normal: {normal_count:,} | Saldiri: {attack_count:,}")
    return df, label_col


def load_feature_names() -> list:
    """Modelin beklediği 42 özellik adını yükle."""
    p = Path(__file__).parent / "models" / "feature_names_unsw.txt"
    if p.exists():
        names = [l.strip() for l in p.read_text().splitlines() if l.strip()]
        log.info(f"Feature names yuklendi: {len(names)} sutun")
        return names
    return []

EXPECTED_FEATURES = load_feature_names()

def build_feature_df(rows: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Dataset satirlarini predictor'in anlayacagi formata donustur."""
    meta_cols = {"id", "label", "attack_cat", "Label", "proto",
                 "service", "state", "srcip", "dstip", "sport", "dsport"}

    # Sadece sayısal feature kolonları al
    feature_cols = [c for c in rows.columns
                    if c not in meta_cols and rows[c].dtype in ["float64","int64","float32","int32"]]
    feat_df = rows[feature_cols].copy()

    # Modelin beklediği 42 sütuna hizala — eksikleri 0 ile doldur
    if EXPECTED_FEATURES:
        for col in EXPECTED_FEATURES:
            if col not in feat_df.columns:
                feat_df[col] = 0.0
        feat_df = feat_df[EXPECTED_FEATURES]

    # Gercekci IP ve port ekle
    n = len(rows)
    feat_df["src_ip"]   = [random.choice(EXTERNAL_IPS + INTERNAL_IPS) for _ in range(n)]
    feat_df["dst_ip"]   = [random.choice(INTERNAL_IPS) for _ in range(n)]
    feat_df["src_port"] = [random.randint(1024, 65535) for _ in range(n)]
    feat_df["dst_port"] = [random.choice([80, 443, 22, 53, 8080, 3306]) for _ in range(n)]
    feat_df["timestamp"] = [datetime.now().isoformat() for _ in range(n)]

    # Gercek label'i koru (dogruluk karsilastirmasi icin)
    if label_col and label_col in rows.columns:
        feat_df["_true_label"] = rows[label_col].values

    return feat_df


def write_alerts(results: pd.DataFrame, true_labels=None):
    """Tahmin sonuclarini alerts.jsonl'e yaz."""
    try:
        with open(ALERTS_FILE, "a", encoding="utf-8") as f:
            for i, row in results.iterrows():
                true_lbl = int(true_labels.iloc[i]) if true_labels is not None else -1
                record = {
                    "ts":          datetime.now().isoformat(),
                    "src_ip":      str(row.get("src_ip", "0.0.0.0")),
                    "dst_ip":      str(row.get("dst_ip", "0.0.0.0")),
                    "src_port":    int(row.get("src_port", 0)),
                    "dst_port":    int(row.get("dst_port", 0)),
                    "is_attack":   bool(row.get("is_attack", False)),
                    "true_label":  true_lbl,
                    "lgbm":        int(row.get("lgbm_pred", 0)),
                    "cnn":         int(row.get("cnn_pred", 0)),
                    "lstm":        int(row.get("lstm_pred", 0)),
                    "mode":        str(row.get("mode", "ensemble")),
                    "latency_ms":  float(row.get("latency_ms", 0)),
                    "source":      "dataset_replay",
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        log.error(f"Alert yazma hatasi: {e}")


# ── Ana simulasyon dongusu ─────────────────────────────────────────────────────

def simulate(df: pd.DataFrame, label_col: str, predictor: Predictor):
    indices = list(range(len(df)))
    random.shuffle(indices)          # Her cevrimi karistir
    pos = 0
    total_sent = 0
    total_attacks = 0
    cycle = 0

    log.info("=" * 55)
    log.info("  DATASET REPLAY SIMULASYONU BASLADI")
    log.info(f"  Mod: {IDS_MODE} | Batch: {BATCH_SIZE} kayit / {BATCH_DELAY}s")
    log.info("=" * 55)

    while not _stop.is_set():
        # Batch al
        if pos + BATCH_SIZE > len(indices):
            # Dataset bitti, yeniden karistir
            cycle += 1
            random.shuffle(indices)
            pos = 0
            log.info(f"Dataset yeniden basladi (Cevrim {cycle})")

        batch_idx = indices[pos : pos + BATCH_SIZE]
        pos += BATCH_SIZE

        batch_rows = df.iloc[batch_idx].reset_index(drop=True)
        true_labels = batch_rows[label_col] if label_col else None

        # Feature DataFrame olustur
        feat_df = build_feature_df(batch_rows, label_col)

        # Dashboard'dan mod değişikliği kontrol et
        get_current_mode(predictor)

        # Tahmin yap
        try:
            results = predictor.predict(feat_df)
            if not results.empty:
                write_alerts(results, true_labels)
                attacks = int(results["is_attack"].sum())
                total_sent    += len(results)
                total_attacks += attacks

                if attacks > 0:
                    log.warning(f"[ALARM] {attacks}/{len(results)} saldiri | Toplam: {total_attacks}/{total_sent}")
                else:
                    log.info(f"[NORMAL] {len(results)} akis | Toplam: {total_sent:,} islem")
        except Exception as e:
            log.error(f"Tahmin hatasi: {e}")

        # Gercekci bekleme (biraz rastgelelik ekle)
        wait = BATCH_DELAY + random.uniform(-1.0, 1.5)
        _stop.wait(max(1.0, wait))

    log.info(f"Simulator durduruldu. Toplam: {total_sent:,} akis, {total_attacks:,} saldiri")


# ── Giris noktasi ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Predictor yukleniyor...")
    predictor = Predictor(mode=IDS_MODE)

    result = load_dataset()
    if isinstance(result, tuple):
        df, label_col = result
    else:
        log.error("Dataset yuklenemedi, cikiliyor.")
        exit(1)

    if df.empty:
        log.error("Dataset bos, cikiliyor.")
        exit(1)

    simulate(df, label_col, predictor)
