# -*- coding: utf-8 -*-
"""
ENSEMBLE MODEL - UNSW-NB15 Dataset
4 modeli (LSTM, LightGBM, XGBoost, CNN) birlestirir
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EnsembleDetectorUNSW:
    def __init__(self):
        self.lstm_model = None
        self.cnn_model = None
        self.lgbm_model = None
        self.xgb_model = None
        self.feature_names = []
        
        with open('processed_data_unsw/feature_names.txt', 'r') as f:
            self.feature_names = [line.strip() for line in f.readlines()]
        
        self.weights = {'lstm': 0.25, 'lightgbm': 0.30, 'xgboost': 0.30, 'cnn': 0.15}
        
        print("[*] UNSW-NB15 Ensemble Detector baslatiliyor...")
        print(f"[*] Model agirliklari: {self.weights}")
    
    def load_models(self):
        print("\n[*] Modeller yukleniyor...")
        
        try:
            self.lstm_model = keras.models.load_model('models/best_lstm_model_unsw.keras')
            print("  [OK] LSTM yuklendi")
        except Exception as e:
            print(f"  [WARN] LSTM yuklenemedi: {e}")
            self.weights['lstm'] = 0
        
        try:
            self.cnn_model = keras.models.load_model('models/best_cnn_model_unsw.keras')
            print("  [OK] CNN yuklendi")
        except Exception as e:
            print(f"  [WARN] CNN yuklenemedi: {e}")
            self.weights['cnn'] = 0
        
        try:
            self.lgbm_model = lgb.Booster(model_file='models/lightgbm_model_unsw.txt')
            print("  [OK] LightGBM yuklendi")
        except Exception as e:
            print(f"  [WARN] LightGBM yuklenemedi: {e}")
            self.weights['lightgbm'] = 0
        
        try:
            self.xgb_model = xgb.Booster()
            self.xgb_model.load_model('models/xgboost_model_unsw.json')
            print("  [OK] XGBoost yuklendi")
        except Exception as e:
            print(f"  [WARN] XGBoost yuklenemedi: {e}")
            self.weights['xgboost'] = 0
        
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
            print(f"\n[*] Normalize edilmis agirliklar: {self.weights}")
        else:
            raise Exception("[ERROR] Hicbir model yuklenemedi!")
    
    def predict_proba(self, X, X_seq):
        predictions = []
        weights_used = []
        
        if self.lstm_model and self.weights['lstm'] > 0:
            lstm_pred = self.lstm_model.predict(X_seq, verbose=0).flatten()
            predictions.append(lstm_pred)
            weights_used.append(self.weights['lstm'])
        
        if self.cnn_model and self.weights['cnn'] > 0:
            X_cnn = X_seq.reshape(X_seq.shape[0], X_seq.shape[1], X_seq.shape[2], 1)
            cnn_pred = self.cnn_model.predict(X_cnn, verbose=0).flatten()
            predictions.append(cnn_pred)
            weights_used.append(self.weights['cnn'])
        
        if self.lgbm_model and self.weights['lightgbm'] > 0:
            lgbm_pred = self.lgbm_model.predict(X)
            predictions.append(lgbm_pred)
            weights_used.append(self.weights['lightgbm'])
        
        if self.xgb_model and self.weights['xgboost'] > 0:
            dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
            xgb_pred = self.xgb_model.predict(dmatrix)
            predictions.append(xgb_pred)
            weights_used.append(self.weights['xgboost'])
        
        weights_used = np.array(weights_used)
        weights_used = weights_used / weights_used.sum()
        
        ensemble_pred = np.zeros(len(predictions[0]))
        for pred, weight in zip(predictions, weights_used):
            ensemble_pred += pred * weight
        
        return ensemble_pred
    
    def evaluate(self, X, X_seq, y_true):
        print("\n[*] Ensemble model degerlendiriliyor...")
        
        y_proba = self.predict_proba(X, X_seq)
        y_pred = (y_proba >= 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_proba)
        }
        
        print("\n" + "="*60)
        print("*** UNSW-NB15 ENSEMBLE PERFORMANSI ***")
        print("="*60)
        print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
        print(f"  Precision: {metrics['precision']*100:.2f}%")
        print(f"  Recall:    {metrics['recall']*100:.2f}%")
        print(f"  F1-Score:  {metrics['f1_score']*100:.2f}%")
        print(f"  AUC:       {metrics['auc']:.4f}")
        print("="*60)
        
        return metrics, y_pred, y_proba

def main():
    print("="*80)
    print("*** ENSEMBLE MODEL - UNSW-NB15 ***")
    print("="*80)
    
    # Veriyi yukle
    print("\n[*] Veri yukleniyor...")
    X_test_seq = np.load('lstm_data_unsw/X_test_seq.npy')
    y_test_seq = np.load('lstm_data_unsw/y_test_seq.npy')
    X_test = np.load('processed_data_unsw/X_test.npy')
    
    seq_len = len(X_test_seq)
    X_test = X_test[:seq_len]
    
    print(f"  [OK] Tabular: {X_test.shape}, Sequence: {X_test_seq.shape}")
    
    # Ensemble
    ensemble = EnsembleDetectorUNSW()
    ensemble.load_models()
    metrics, y_pred, y_proba = ensemble.evaluate(X_test, X_test_seq, y_test_seq)
    
    # Confusion matrix
    cm = confusion_matrix(y_test_seq, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Saldiri'], yticklabels=['Normal', 'Saldiri'])
    plt.title('UNSW-NB15 - Ensemble Confusion Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('models/ensemble_unsw_confusion_matrix.png', dpi=300)
    print("\n[OK] models/ensemble_unsw_confusion_matrix.png")
    
    # Rapor
    report = f"""
UNSW-NB15 - ENSEMBLE MODEL RAPORU
{'='*80}
Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Performans:
  Accuracy:  {metrics['accuracy']*100:.2f}%
  Precision: {metrics['precision']*100:.2f}%
  Recall:    {metrics['recall']*100:.2f}%
  F1-Score:  {metrics['f1_score']*100:.2f}%
  AUC:       {metrics['auc']:.4f}

Model Agirliklari:
  LSTM: {ensemble.weights.get('lstm', 0)*100:.0f}%
  LightGBM: {ensemble.weights.get('lightgbm', 0)*100:.0f}%
  XGBoost: {ensemble.weights.get('xgboost', 0)*100:.0f}%
  CNN: {ensemble.weights.get('cnn', 0)*100:.0f}%
"""
    
    with open('models/ensemble_unsw_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("[OK] models/ensemble_unsw_report.txt")
    print("\n[SUCCESS] UNSW-NB15 Ensemble tamamlandi!")

if __name__ == "__main__":
    main()
