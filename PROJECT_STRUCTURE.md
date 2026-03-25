# 📂 SİBER SALDIRI TESPİT PROJESİ - GERÇEK KLASÖR YAPISI

## 🗂️ Proje Klasör Organizasyonu

```
siber_saldırı_project/
│
├── 📁 datasets/                          # Ham veri setleri
│   ├── KDDTrain+.txt (19.1 MB)
│   └── KDDTest+.txt (3.4 MB)
│
├── 📁 processed_data/                    # İşlenmiş veri
│   ├── X_train.npy (41.3 MB)
│   ├── y_train.npy (1.0 MB)
│   ├── X_test.npy (7.4 MB)
│   ├── y_test.npy (180 KB)
│   └── feature_names.txt (644 bytes)
│
├── 📁 lstm_data/                         # LSTM formatında sequence veri
│   ├── X_train_seq.npy
│   ├── y_train_seq.npy
│   ├── X_test_seq.npy
│   ├── y_test_seq.npy
│   └── metadata.npy
│
├── 📁 models/                            # Eğitilmiş modeller ve grafikler (21 dosya)
│   ├── 🤖 Modeller (4):
│   │   ├── best_lstm_model.keras (1.7 MB)
│   │   ├── best_cnn_model.keras (974 KB)
│   │   ├── lightgbm_model.txt (544 KB)
│   │   ├── xgboost_model.json (643 KB)
│   │   └── xgboost_model_clean.json (642 KB)
│   │
│   ├── 📊 Görselleştirmeler (13):
│   │   ├── training_history.png (LSTM)
│   │   ├── confusion_matrix.png (LSTM)
│   │   ├── cnn_training_history.png (CNN)
│   │   ├── cnn_confusion_matrix.png (CNN)
│   │   ├── lgbm_feature_importance.png
│   │   ├── lgbm_roc_curve.png
│   │   ├── lgbm_confusion_matrix.png
│   │   ├── xgb_feature_importance.png
│   │   ├── xgb_roc_curve.png
│   │   ├── xgb_confusion_matrix.png
│   │   ├── model_comparison.png (LSTM vs LightGBM)
│   │   └── 4_model_comprehensive_comparison.png (1.1 MB) ⭐ Kapsamlı
│   │
│   ├── 📄 Raporlar (3):
│   │   ├── comparison_report.txt
│   │   └── 4_Model_Detailed_Comparison_Report.pdf (11.5 KB) ⭐ Detaylı
│   │
│   └── 📋 Feature Importance CSV (2):
│       ├── lightgbm_model_feature_importance.csv
│       └── xgboost_model_feature_importance.csv
│
├── 📁 .venv/                             # Python virtual environment
│
├── 🐍 Python Scriptleri (13):
│   ├── data_preprocessing.py (12.2 KB)         # Veri ön işleme
│   ├── prepare_lstm_data.py (5.5 KB)          # LSTM veri hazırlama
│   ├── lstm_model.py (13.7 KB)                # LSTM model eğitimi
│   ├── cnn_model.py (14.2 KB)                 # CNN model eğitimi
│   ├── lightgbm_model.py (13.4 KB)            # LightGBM model eğitimi
│   ├── xgboost_model.py (15.4 KB)             # XGBoost model eğitimi
│   ├── compare_models.py (14.3 KB)            # 2 model karşılaştırma
│   ├── test_model.py (8.7 KB)                 # Model test scripti
│   ├── create_comparison_chart.py (11.7 KB)   # 4 model PNG grafiği
│   ├── create_pdf_report.py (15.0 KB)         # PDF rapor oluşturma
│   ├── fix_xgb_model.py (447 bytes)           # XGBoost model fix
│   └── show_structure.py (5.1 KB)             # Klasör yapısı gösterimi
│
└── 📄 Dokümantasyon (3):
    ├── README.md (10.9 KB)                     # Ana dokümantasyon
    ├── FINAL_REPORT.md (6.0 KB)               # Final proje raporu
    └── requirements.txt (123 bytes)            # Python bağımlılıkları
```

---

## 📊 Dosya İstatistikleri

| Kategori | Sayı | Toplam Boyut |
|----------|------|--------------|
| **Python Scriptleri** | 13 | ~138 KB |
| **Eğitilmiş Modeller** | 5 | ~4.0 MB |
| **Görselleştirmeler** | 13 | ~2.5 MB |
| **Veri Dosyaları** | 10 | ~72 MB |
| **Dokümantasyon** | 4 | ~29 KB |
| **TOPLAM** | 45+ dosya | ~78 MB |

---

## 🎯 Dosya Grupları

### 1. **Veri İşleme Pipeline**
```
datasets/KDDTrain+.txt
    ↓ (data_preprocessing.py)
processed_data/X_train.npy, y_train.npy
    ↓ (prepare_lstm_data.py) [sadece LSTM/CNN için]
lstm_data/X_train_seq.npy
```

### 2. **Model Eğitimi**
- `lstm_model.py` → `models/best_lstm_model.keras`
- `cnn_model.py` → `models/best_cnn_model.keras`
- `lightgbm_model.py` → `models/lightgbm_model.txt`
- `xgboost_model.py` → `models/xgboost_model.json`

### 3. **Analiz ve Raporlama**
- `compare_models.py` → `models/model_comparison.png`
- `create_comparison_chart.py` → `models/4_model_comprehensive_comparison.png`
- `create_pdf_report.py` → `models/4_Model_Detailed_Comparison_Report.pdf`

### 4. **Test ve Deployment**
- `test_model.py` - Modelleri test etme
- `models/*.keras, *.txt, *.json` - Production-ready modeller

---

## 🚀 Hızlı Başlangıç

### Tüm Veri Pipeline'ı Çalıştırma:
```bash
# 1. Veri hazırlama
python data_preprocessing.py

# 2. LSTM/CNN için sequence oluşturma
python prepare_lstm_data.py

# 3. Modelleri eğitme
python lstm_model.py
python cnn_model.py
python lightgbm_model.py
python xgboost_model.py

# 4. Karşılaştırma ve raporlar
python create_comparison_chart.py
python create_pdf_report.py
```

### Sadece Testleme:
```bash
# Eğitilmiş modeli test et
python test_model.py
```

---

## 📦 Proje Bağımlılıkları

`requirements.txt` içeriği:
```
tensorflow>=2.0.0
scikit-learn>=1.0.0
lightgbm>=3.0.0
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.3.0
seaborn>=0.11.0
reportlab>=3.0.0
```

Kurulum:
```bash
pip install -r requirements.txt
```

---

## 💡 Önemli Notlar

1. **Veri Klasörleri:**
   - `datasets/` - Ham veri (KDD Cup 1999)
   - `processed_data/` - Normalize edilmiş, encoded veri
   - `lstm_data/` - Sequence formatında veri (sadece LSTM/CNN için)

2. **Model Dosyaları:**
   - `.keras` - TensorFlow/Keras modelleri (LSTM, CNN)
   - `.txt` - LightGBM modeli
   - `.json` - XGBoost modeli

3. **Görselleştirmeler:**
   - Her model için ayrı confusion matrix ve grafikler
   - Karşılaştırma grafikleri
   - Feature importance grafikleri (tree-based modeller)

4. **Raporlar:**
   - **PNG:** `4_model_comprehensive_comparison.png` - Görsel karşılaştırma
   - **PDF:** `4_Model_Detailed_Comparison_Report.pdf` - Detaylı tablo raporu
   - **Markdown:** `README.md`, `FINAL_REPORT.md` - Dokümantasyon

---

## ✅ Proje Durumu

**Tamamlanma:** %100  
**Durum:** Production Ready  
**Son Güncelleme:** 30 Aralık 2025  
**Toplam Modeller:** 4 (LSTM, CNN, LightGBM, XGBoost)  
**Toplam Scriptler:** 13  
**Toplam Görselleştirme:** 13+  

---

**Geliştirici:** Nefise  
**Proje:** Siber Saldırı Tespit Sistemi  
**Teknoloji Stack:** Python 3.8+, TensorFlow 2.0+, LightGBM, XGBoost
