# -*- coding: utf-8 -*-
"""
XGBoost Model - UNSW-NB15 Dataset
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time

print("="*80)
print("XGBOOST MODEL - UNSW-NB15 DATASET")
print("="*80)

start_time = time.time()
X_train = np.load('processed_data_unsw/X_train.npy')
y_train = np.load('processed_data_unsw/y_train.npy')
X_test = np.load('processed_data_unsw/X_test.npy')
y_test = np.load('processed_data_unsw/y_test.npy')

with open('processed_data_unsw/feature_names.txt', 'r') as f:
    feature_names = [line.strip() for line in f.readlines()]

print(f"  [OK] Train: {X_train.shape}, Test: {X_test.shape}")

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

params = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'],
    'tree_method': 'hist',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'seed': 42
}

print("\n" + "="*80)
print("EGITIM BASLIYOR")
print("="*80)

evals = [(dtrain, 'train'), (dtest, 'valid')]
model = xgb.train(
    params, dtrain,
    num_boost_round=1000,
    evals=evals,
    early_stopping_rounds=50,
    verbose_eval=50
)

model.save_model('models/xgboost_model_unsw.json')

y_pred_proba = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
y_pred = (y_pred_proba >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "="*80)
print("UNSW-NB15 - XGBOOST PERFORMANS")
print("="*80)
print(f"  Accuracy:  {acc*100:.2f}%")
print(f"  Precision: {prec*100:.2f}%")
print(f"  Recall:    {rec*100:.2f}%")
print(f"  F1-Score:  {f1*100:.2f}%")
print(f"  AUC:       {auc:.4f}")
print(f"  Egitim: {time.time()-start_time:.2f}s")
print("="*80)

# Feature importance
importance = model.get_score(importance_type='gain')
feature_imp = pd.DataFrame(list(importance.items()), columns=['feature', 'importance']).sort_values('importance', ascending=False)
feature_imp.to_csv('models/xgboost_unsw_feature_importance.csv', index=False)

plt.figure(figsize=(12, 8))
plt.barh(feature_imp['feature'][:20][::-1], feature_imp['importance'][:20][::-1], color='#e74c3c')
plt.title('UNSW-NB15 - XGBoost Feature Importance')
plt.tight_layout()
plt.savefig('models/xgboost_unsw_feature_importance.png', dpi=300)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['Normal', 'Saldiri'], yticklabels=['Normal', 'Saldiri'])
plt.title('UNSW-NB15 - XGBoost Confusion Matrix')
plt.tight_layout()
plt.savefig('models/xgboost_unsw_confusion_matrix.png', dpi=300)

# ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, linewidth=2, color='#e74c3c', label=f'AUC = {auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('UNSW-NB15 - XGBoost ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig('models/xgboost_unsw_roc_curve.png', dpi=300)

print("\n[SUCCESS] UNSW-NB15 XGBoost tamamlandi!")
