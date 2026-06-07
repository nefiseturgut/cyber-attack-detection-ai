#!/usr/bin/env python3
"""
IDS Predictor
Eğitilmiş modelleri yükler, gelen feature DataFrame'ini tahmin eder.
Mod: "ensemble" | "fast" (LightGBM) | "accurate" (CNN)
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger("IDS_PRED")

MODELS_DIR   = Path(__file__).parent / "models"
META_COLS    = {"src_ip", "dst_ip", "src_port", "dst_port", "timestamp"}
ATTACK_TYPES = {
    (0, 0, 0): "Normal",
    (1, 0, 0): "Saldırı (LightGBM)",
    (0, 1, 0): "Saldırı (CNN)",
    (0, 0, 1): "Saldırı (LSTM)",
    (1, 1, 0): "Saldırı (LGB+CNN)",
    (1, 0, 1): "Saldırı (LGB+LSTM)",
    (0, 1, 1): "Saldırı (CNN+LSTM)",
    (1, 1, 1): "Saldırı (Ensemble)",
}


class Predictor:
    def __init__(self, mode: str = "ensemble"):
        """
        mode: "ensemble" | "fast" | "accurate"
        """
        self.mode         = mode
        self.feature_cols = self._load_feature_names()
        self.scaler       = self._load_scaler()
        self.lgbm         = self._load_lgbm()
        self.cnn          = self._load_keras("cnn")
        self.lstm         = self._load_keras("lstm")

        loaded = [n for n, m in [("LightGBM", self.lgbm), ("CNN", self.cnn), ("LSTM", self.lstm)] if m]
        log.info(f"Predictor hazır — mod={mode} — yüklenen: {loaded}")

    # ── Yükleme ───────────────────────────────────────────────────────────────

    def _load_feature_names(self) -> list:
        p = MODELS_DIR / "feature_names_unsw.txt"
        if p.exists():
            names = [l.strip() for l in p.read_text().splitlines() if l.strip()]
            log.info(f"Feature names yüklendi: {len(names)} sütun")
            return names
        log.warning("feature_names_unsw.txt bulunamadı — ham sütunlar kullanılacak")
        return []

    def _load_scaler(self):
        p = MODELS_DIR / "scaler_unsw.pkl"
        if not p.exists():
            return None
        try:
            import pickle
            with open(p, "rb") as f:
                scaler = pickle.load(f)
            log.info("Scaler yüklendi")
            return scaler
        except Exception as e:
            log.warning(f"Scaler yüklenemedi: {e}")
            return None

    def _load_lgbm(self):
        p = MODELS_DIR / "lgbm_unsw.txt"
        if not p.exists():
            return None
        try:
            import lightgbm as lgb
            m = lgb.Booster(model_file=str(p))
            log.info("LightGBM yüklendi")
            return m
        except Exception as e:
            log.warning(f"LightGBM yüklenemedi: {e}")
            return None

    def _load_keras(self, name: str):
        for ext in [".keras", ".h5"]:
            p = MODELS_DIR / f"{name}_unsw{ext}"
            if p.exists():
                try:
                    import tensorflow as tf
                    m = tf.keras.models.load_model(str(p))
                    log.info(f"{name.upper()} yüklendi ({p.name})")
                    return m
                except Exception as e:
                    log.warning(f"{name.upper()} yüklenemedi: {e}")
        return None

    # ── Ön işleme ─────────────────────────────────────────────────────────────

    def _prepare(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        numeric = df.drop(columns=[c for c in META_COLS if c in df.columns], errors="ignore")
        numeric = numeric.select_dtypes(include=[np.number])

        if self.feature_cols:
            for col in self.feature_cols:
                if col not in numeric.columns:
                    numeric[col] = 0.0
            numeric = numeric[self.feature_cols]

        numeric = numeric.fillna(0.0).replace([np.inf, -np.inf], 0.0)

        # Live trafik verisini 0-1 arasına normalize et (log scale)
        from sklearn.preprocessing import MinMaxScaler
        try:
            vals = numeric.values.astype(float)
            # Log1p normalize — ağ trafiği değerleri çok geniş aralıkta
            vals = np.log1p(np.abs(vals))
            col_max = vals.max(axis=0)
            col_max[col_max == 0] = 1
            vals = vals / col_max
            return vals
        except Exception as e:
            log.warning(f"Normalize hatası: {e}")
            return numeric.values

    # ── Tahmin ────────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        t0  = time.time()
        X   = self._prepare(df)
        if X is None or len(X) == 0:
            return pd.DataFrame()

        results = df[list(META_COLS & set(df.columns))].copy().reset_index(drop=True)

        lgb_pred  = self._pred_lgbm(X)
        cnn_pred  = self._pred_keras(self.cnn,  X, "CNN")
        lstm_pred = self._pred_keras(self.lstm, X, "LSTM")

        if self.mode == "fast":
            final = lgb_pred if lgb_pred is not None else cnn_pred
        elif self.mode == "accurate":
            final = cnn_pred if cnn_pred is not None else lgb_pred
        else:  # ensemble
            votes = np.zeros(len(X), dtype=int)
            preds = []
            for p in [lgb_pred, cnn_pred, lstm_pred]:
                if p is not None:
                    votes += p.astype(int)
                    preds.append(p.astype(int))
            n_models = len(preds) if preds else 1
            final = (votes >= (n_models / 2)).astype(int)

        results["prediction"]   = final if final is not None else 0
        results["is_attack"]    = results["prediction"].astype(bool)
        results["lgbm_pred"]    = lgb_pred  if lgb_pred  is not None else np.zeros(len(X))
        results["cnn_pred"]     = cnn_pred  if cnn_pred  is not None else np.zeros(len(X))
        results["lstm_pred"]    = lstm_pred if lstm_pred is not None else np.zeros(len(X))
        results["latency_ms"]   = round((time.time() - t0) * 1000, 2)
        results["mode"]         = self.mode

        attacks = int(results["is_attack"].sum())
        log.info(f"Tahmin: {len(results)} akış — {attacks} saldırı — {results['latency_ms'].iloc[0]:.1f}ms")
        return results

    def _pred_lgbm(self, X: np.ndarray) -> Optional[np.ndarray]:
        if self.lgbm is None:
            return None
        try:
            proba = self.lgbm.predict(X, num_iteration=self.lgbm.best_iteration)
            return (proba >= 0.5).astype(int)
        except Exception as e:
            log.error(f"LightGBM predict hatası: {e}")
            return None

    def _pred_keras(self, model, X: np.ndarray, name: str) -> Optional[np.ndarray]:
        if model is None:
            return None
        try:
            inp = X.reshape(len(X), X.shape[1], 1) if name == "CNN" else X.reshape(len(X), 1, X.shape[1])
            proba = model.predict(inp, verbose=0).flatten()
            return (proba >= 0.5).astype(int)
        except Exception as e:
            log.error(f"{name} predict hatası: {e}")
            return None

    def switch_mode(self, mode: str):
        assert mode in ("ensemble", "fast", "accurate"), f"Geçersiz mod: {mode}"
        self.mode = mode
        log.info(f"Mod değişti → {mode}")
