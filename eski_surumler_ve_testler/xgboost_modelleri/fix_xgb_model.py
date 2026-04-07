"""
XGBoost modelini feature names olmadan yeniden kaydet
"""

import numpy as np
import xgboost as xgb

# Modeli yükle
model = xgb.Booster()
model.load_model('models/xgboost_model.json')

# Feature names'i temizle
model.feature_names = None
model.feature_types = None

# Yeniden kaydet
model.save_model('models/xgboost_model_clean.json')

print("✅ XGBoost modeli feature names olmadan kaydedildi: xgboost_model_clean.json")
