# -*- coding: utf-8 -*-
"""
LightGBM Model - UNSW-NB15 Dataset
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time

print("="*80)
print("LIGHTGBM MODEL - UNSW-NB15 DATASET")
print("="*80)

start_time = time.time()
X_train = np.load('processed_data_unsw/X_train.npy')
y_train = np.load('processed_data_unsw/y_train.npy')
X_test = np.load('processed_data_unsw/X_test.npy')
y_test = np.load('processed_data_unsw/y_test.npy')

with open('processed_data_unsw/feature_names.txt', 'r') as f:
    feature_names = [line.strip() for line in f.readlines()]

print(f"  [OK] Train: {X_train.shape}, Test: {X_test.shape}")

lgb_train = lgb.Dataset(X_train, y_train, feature_name=feature_names)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train, feature_name=feature_names)

params = {
    'objective': 'binary',
    'metric': ['binary_logloss', 'auc'],
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 1
}

print("\n" + "="*80)
print("EGITIM BASLIYOR")
print("="*80)

model = lgb.train(
    params, lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_train, lgb_test],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
)

model.save_model('models/lightgbm_model_unsw.txt')

y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = (y_pred_proba >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "="*80)
print("UNSW-NB15 - LIGHTGBM PERFORMANS")
print("="*80)
print(f"  Accuracy:  {acc*100:.2f}%")
print(f"  Precision: {prec*100:.2f}%")
print(f"  Recall:    {rec*100:.2f}%")
print(f"  F1-Score:  {f1*100:.2f}%")
print(f"  AUC:       {auc:.4f}")
print(f"  Egitim: {time.time()-start_time:.2f}s")
print("="*80)

# Feature importance
importance = model.feature_importance(importance_type='gain')
feature_imp = pd.DataFrame({'feature': feature_names, 'importance': importance}).sort_values('importance', ascending=False)
feature_imp.to_csv('models/lightgbm_unsw_feature_importance.csv', index=False)

plt.figure(figsize=(12, 8))
plt.barh(feature_imp['feature'][:20][::-1], feature_imp['importance'][:20][::-1])
plt.title('UNSW-NB15 - LightGBM Feature Importance')
plt.tight_layout()
plt.savefig('models/lightgbm_unsw_feature_importance.png', dpi=300)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Normal', 'Saldiri'], yticklabels=['Normal', 'Saldiri'])
plt.title('UNSW-NB15 - LightGBM Confusion Matrix')
plt.tight_layout()
plt.savefig('models/lightgbm_unsw_confusion_matrix.png', dpi=300)

# ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('UNSW-NB15 - LightGBM ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig('models/lightgbm_unsw_roc_curve.png', dpi=300)

print("\n[SUCCESS] UNSW-NB15 LightGBM tamamlandi!")
