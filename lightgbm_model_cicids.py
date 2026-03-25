# -*- coding: utf-8 -*-
"""
LightGBM Model - CICIDS2018 Dataset
Gradient Boosting ile Siber Saldiri Tespiti
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("LIGHTGBM MODEL - CICIDS2018 DATASET")
print("="*80)

# Veriyi yukle
print("\n[*] CICIDS2018 verisi yukleniyor...")
start_time = time.time()

X_train = np.load('processed_data_cicids/X_train.npy')
y_train = np.load('processed_data_cicids/y_train.npy')
X_test = np.load('processed_data_cicids/X_test.npy')
y_test = np.load('processed_data_cicids/y_test.npy')

# Feature names
with open('processed_data_cicids/feature_names.txt', 'r') as f:
    feature_names = [line.strip() for line in f.readlines()]

load_time = time.time() - start_time
print(f"  [OK] Veri yukleme suresi: {load_time:.2f} saniye")
print(f"  [OK] Train data: {X_train.shape}")
print(f"  [OK] Test data: {X_test.shape}")
print(f"  [OK] Features: {len(feature_names)}")

# Sinif dagilimi
unique, counts = np.unique(y_train, return_counts=True)
print(f"\n[*] Sinif dagilimi (Train):")
for label, count in zip(unique, counts):
    print(f"  Class {label}: {count:,} samples ({count/len(y_train)*100:.1f}%)")

# LightGBM dataset olustur
print("\n[*] LightGBM dataset olusturuluyor...")
lgb_train = lgb.Dataset(X_train, y_train, feature_name=feature_names)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train, feature_name=feature_names)

# Model parametreleri
params = {
    'objective': 'binary',
    'metric': ['binary_logloss', 'auc'],
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 1,
    'max_depth': -1,
    'min_child_samples': 20,
    'num_threads': -1
}

print("\n[*] Model parametreleri:")
for key, value in params.items():
    print(f"  {key}: {value}")

# Model egitimi
print("\n" + "="*80)
print("MODEL EGITIMI BASLIYOR")
print("="*80)
print(f"Baslangic: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

train_start = time.time()

model = lgb.train(
    params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_train, lgb_test],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50)
    ]
)

train_time = time.time() - train_start
print(f"\nBitis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Egitim suresi: {train_time:.2f} saniye ({train_time/60:.2f} dakika)")
print("="*80)

# Modeli kaydet
print("\n[*] Model kaydediliyor...")
model.save_model('models/lightgbm_model_cicids.txt')
print("  [OK] models/lightgbm_model_cicids.txt")

# Tahmin yap
print("\n[*] Tahminler yapiliyor...")
pred_start = time.time()
y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = (y_pred_proba >= 0.5).astype(int)
pred_time = time.time() - pred_start

print(f"  [OK] Tahmin suresi: {pred_time:.4f} saniye")
print(f"  [OK] Ortalama tahmin suresi: {pred_time/len(X_test)*1000:.4f} ms/sample")

# Metrikler
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "="*80)
print("CICIDS2018 - LIGHTGBM MODEL PERFORMANSI")
print("="*80)
print(f"  Accuracy:  {accuracy*100:.2f}%")
print(f"  Precision: {precision*100:.2f}%")
print(f"  Recall:    {recall*100:.2f}%")
print(f"  F1-Score:  {f1*100:.2f}%")
print(f"  AUC:       {auc:.4f}")
print(f"  Egitim suresi: {train_time:.2f}s")
print(f"  Tahmin suresi: {pred_time:.4f}s")
print("="*80)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n[*] Confusion Matrix:")
print(cm)

# Feature Importance
print("\n[*] Feature importance hesaplaniyor...")
importance = model.feature_importance(importance_type='gain')
feature_imp = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False)

# Top 20 feature
top_features = feature_imp.head(20)
print("\nTop 20 En Onemli Ozellikler:")
print(top_features.to_string(index=False))

# Feature importance CSV
feature_imp.to_csv('models/lightgbm_cicids_feature_importance.csv', index=False)
print("\n[OK] models/lightgbm_cicids_feature_importance.csv")

# Gorselestirme 1: Feature Importance
print("\n[*] Grafikler olusturuluyor...")
plt.figure(figsize=(12, 8))
plt.barh(top_features['feature'][::-1], top_features['importance'][::-1])
plt.xlabel('Importance (Gain)', fontsize=12, fontweight='bold')
plt.ylabel('Feature', fontsize=12, fontweight='bold')
plt.title('CICIDS2018 - LightGBM Top 20 Feature Importance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('models/lightgbm_cicids_feature_importance.png', dpi=300, bbox_inches='tight')
print("  [OK] models/lightgbm_cicids_feature_importance.png")

# Gorselestirme 2: Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Normal', 'Saldiri'],
            yticklabels=['Normal', 'Saldiri'],
            cbar_kws={'label': 'Sample Count'})
plt.title('CICIDS2018 - LightGBM Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Gercek Sinif', fontsize=13, fontweight='bold')
plt.xlabel('Tahmin Edilen Sinif', fontsize=13, fontweight='bold')

tn, fp, fn, tp = cm.ravel()
total = tn + fp + fn + tp
stats_text = f"""
Toplam: {total:,} samples

True Negatives:  {tn:,} ({tn/total*100:.1f}%)
False Positives: {fp:,} ({fp/total*100:.1f}%)
False Negatives: {fn:,} ({fn/total*100:.1f}%)
True Positives:  {tp:,} ({tp/total*100:.1f}%)
"""
plt.gcf().text(0.02, 0.02, stats_text, fontsize=10,
               family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.tight_layout()
plt.savefig('models/lightgbm_cicids_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("  [OK] models/lightgbm_cicids_confusion_matrix.png")

# Gorselestirme 3: ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, linewidth=2, label=f'LightGBM (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title('CICIDS2018 - LightGBM ROC Curve', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('models/lightgbm_cicids_roc_curve.png', dpi=300, bbox_inches='tight')
print("  [OK] models/lightgbm_cicids_roc_curve.png")

# Rapor olustur
report_text = f"""
{'='*80}
CICIDS2018 - LIGHTGBM MODEL RAPORU
{'='*80}

Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

1. DATASET BILGILERI
{'='*80}
Dataset: CICIDS2018
Train samples: {len(y_train):,}
Test samples: {len(y_test):,}
Features: {len(feature_names)}

Sinif dagilimi:
  Normal (0): {counts[0]:,} ({counts[0]/len(y_train)*100:.1f}%)
  Saldiri (1): {counts[1]:,} ({counts[1]/len(y_train)*100:.1f}%)

2. MODEL PARAMETRELERI
{'='*80}
Boosting type: GBDT
Number of trees: {model.num_trees()}
Learning rate: 0.05
Max depth: -1 (no limit)
Num leaves: 31
Feature fraction: 0.9
Bagging fraction: 0.8

3. EGITIM BILGILERI
{'='*80}
Egitim suresi: {train_time:.2f} saniye ({train_time/60:.2f} dakika)
Best iteration: {model.best_iteration}
Early stopping: 50 rounds

4. PERFORMANS METRIKLERI
{'='*80}
Test Seti Sonuclari:
  * Accuracy:  {accuracy*100:.2f}%
  * Precision: {precision*100:.2f}%
  * Recall:    {recall*100:.2f}%
  * F1-Score:  {f1*100:.2f}%
  * AUC:       {auc:.4f}

Tahmin Performansi:
  * Toplam tahmin suresi: {pred_time:.4f} saniye
  * Ortalama tahmin suresi: {pred_time/len(X_test)*1000:.4f} ms/sample
  * Saniyede tahmin: {len(X_test)/pred_time:.0f} samples/sec

Confusion Matrix:
  True Negatives:  {tn:,}
  False Positives: {fp:,}
  False Negatives: {fn:,}
  True Positives:  {tp:,}

5. EN ONEMLI OZELLIKLER (Top 10)
{'='*80}
{top_features.head(10).to_string(index=False)}

6. SONUC
{'='*80}
LightGBM modeli CICIDS2018 dataseti uzerinde basariyla egitildi.
Model, {accuracy*100:.2f}% dogruluk orani ile siber saldirilari tespit edebiliyor.

Avantajlar:
  - Cok hizli egitim ({train_time:.2f} saniye)
  - Hizli tahmin ({pred_time/len(X_test)*1000:.4f} ms/sample)
  - Feature importance analizi
  - Yuksek AUC skoru ({auc:.4f})

Kaydedilen dosyalar:
  - models/lightgbm_model_cicids.txt
  - models/lightgbm_cicids_feature_importance.png
  - models/lightgbm_cicids_feature_importance.csv
  - models/lightgbm_cicids_confusion_matrix.png
  - models/lightgbm_cicids_roc_curve.png

{'='*80}
Gelistirici: Nefise
Tarih: {datetime.now().strftime('%d %B %Y')}
{'='*80}
"""

with open('models/lightgbm_cicids_report.txt', 'w', encoding='utf-8') as f:
    f.write(report_text)

print("  [OK] models/lightgbm_cicids_report.txt")

print("\n" + "="*80)
print("[SUCCESS] CICIDS2018 - LIGHTGBM MODEL TAMAMLANDI!")
print("="*80)
print(f"\nToplam sure: {time.time() - start_time:.2f} saniye")
print("\nOlusturulan dosyalar:")
print("  [OK] models/lightgbm_model_cicids.txt")
print("  [OK] models/lightgbm_cicids_feature_importance.png")
print("  [OK] models/lightgbm_cicids_feature_importance.csv")
print("  [OK] models/lightgbm_cicids_confusion_matrix.png")
print("  [OK] models/lightgbm_cicids_roc_curve.png")
print("  [OK] models/lightgbm_cicids_report.txt")
print("\n" + "="*80)
