# 🎯 UNSW-NB15 Model Sonuçları

## 📊 Performans Karşılaştırması

| Model | Accuracy | Precision | Recall | F1-Score | AUC | Eğitim Süresi |
|-------|----------|-----------|--------|----------|-----|---------------|
| **LSTM** | 96.65% | - | - | - | - | ~4 dakika |
| **CNN** | **98.55%** | - | **98.64%** | - | - | ~20 dakika |
| **LightGBM** | 87.70% | 82.38% | 98.80% | 89.84% | 0.9869 | 2.60s |
| **XGBoost** | 87.40% | 82.00% | 98.82% | 89.62% | 0.9853 | 8.14s |
| **Ensemble** | 95.28% | 92.23% | **99.84%** | 95.88% | **0.9984** | ~2 dakika |

## 🏆 Kazananlar

- **En Yüksek Accuracy**: CNN (98.55%)
- **En Yüksek Recall**: Ensemble (99.84%)
- **En Yüksek AUC**: Ensemble (0.9984)
- **En Hızlı Eğitim**: LightGBM (2.60s)

## 📁 Oluşturulan Dosyalar

### Modeller:
- ✅ `best_lstm_model_unsw.keras`
- ✅ `best_cnn_model_unsw.keras`
- ✅ `lightgbm_model_unsw.txt`
- ✅ `xgboost_model_unsw.json`

### Görselleştirmeler:
- ✅ LSTM: `lstm_unsw_training_history.png`, `lstm_unsw_confusion_matrix.png`
- ✅ CNN: `cnn_unsw_training_history.png`, `cnn_unsw_confusion_matrix.png`
- ✅ LightGBM: `lightgbm_unsw_feature_importance.png`, `lightgbm_unsw_confusion_matrix.png`, `lightgbm_unsw_roc_curve.png`
- ✅ XGBoost: `xgboost_unsw_feature_importance.png`, `xgboost_unsw_confusion_matrix.png`, `xgboost_unsw_roc_curve.png`
- ✅ Ensemble: `ensemble_unsw_confusion_matrix.png`, `ensemble_unsw_report.txt`

### Feature Importance:
- ✅ `lightgbm_unsw_feature_importance.csv`
- ✅ `xgboost_unsw_feature_importance.csv`

## 💡 Önemli Gözlemler

1. **CNN Modeli** en yüksek bireysel accuracy'ye sahip (98.55%)
2. **Ensemble Model** en yüksek recall (99.84%) ve AUC (0.9984) sağlıyor
3. **Gradient Boosting modelleri** (LightGBM, XGBoost) çok hızlı ama accuracy daha düşük
4. **LSTM** dengeli bir performans sunuyor (96.65% accuracy)

## 🎯 Sonuç

UNSW-NB15 dataset'i için:
- **Gerçek zamanlı sistemler**: LightGBM (hız öncelikli)
- **Yüksek doğruluk**: CNN veya Ensemble
- **Saldırıları kaçırmama**: Ensemble (99.84% recall)
