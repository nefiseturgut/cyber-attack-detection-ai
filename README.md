# 🛡️ Çok-Dataset Ensemble Siber Saldırı Tespit Sistemi

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

**Kapsamlı bir makine öğrenmesi projesi: 3 Dataset × 4 Model + Ensemble = 15 Eğitilmiş Model**

---

## 📋 Proje Özeti

Bu proje, **3 farklı siber saldırı tespit veri seti** kullanarak **15 farklı makine öğrenmesi modeli** geliştirmiş ve **ensemble yaklaşımı** ile yüksek performanslı siber saldırı tespit sistemleri oluşturmuştur.

### 🎯 Ana Hedef
Farklı veri setleri ve model türlerini karşılaştırarak, en etkili siber saldırı tespit yöntemlerini belirlemek ve gerçek dünya uygulamaları için önerilerde bulunmak.

---

## 🗂️ Kullanılan Veri Setleri

| Dataset | Özellikler | Kayıt Sayısı | Kullanım Amacı |
|---------|-----------|--------------|----------------|
| **KDD Cup 1999** | 41 | ~500,000 | Klasik benchmark |
| **CICIDS2018** | 80 | ~1,000,000 | Modern saldırılar |
| **UNSW-NB15** | 42 | ~257,000 | Dengeli, gerçekçi |

---

## 🤖 Geliştirilen Modeller

Her veri seti için **4 farklı model türü** + **1 ensemble model**:

### 1. LSTM (Long Short-Term Memory)
- 🧠 Derin öğrenme
- 📊 Sequence-based yaklaşım
- ⏱️ Eğitim: ~30-40 dakika

### 2. CNN (Convolutional Neural Network)
- 🧠 Derin öğrenme
- 🔍 Pattern recognition
- ⏱️ Eğitim: ~20-30 dakika

### 3. LightGBM
- ⚡ Gradient Boosting
- 🚀 Çok hızlı (2-3 saniye)
- 💾 Hafif ve verimli

### 4. XGBoost
- ⚡ Gradient Boosting
- 🎯 Yüksek accuracy
- 📈 Feature importance

### 5. Ensemble (Hibrit)
- 🏆 Tüm modellerin kombinasyonu
- 🎭 Weighted voting
- ✨ En yüksek performans

**Toplam: 15 Eğitilmiş Model**

---

## 📊 Performans Sonuçları

### 🏆 En İyi Sonuçlar

| Dataset | En İyi Model | Accuracy | Recall | AUC |
|---------|-------------|----------|--------|-----|
| **KDD Cup 1999** | Ensemble | 82%+ | 70%+ | 0.97+ |
| **CICIDS2018** | CNN/Ensemble | 97%+ | 96%+ | 0.99+ |
| **UNSW-NB15** | CNN | **98.55%** | 98.64% | 0.99 |
| **UNSW-NB15** | Ensemble | 95.28% | **99.84%** | **0.9984** |

### 📈 UNSW-NB15 Detaylı Performans

| Model | Accuracy | Precision | Recall | F1-Score | AUC | Eğitim Süresi |
|-------|----------|-----------|--------|----------|-----|---------------|
| LSTM | 96.65% | 95.00% | 97.00% | 96.00% | 0.9800 | ~4 dk |
| **CNN** | **98.55%** | 97.00% | 98.64% | **97.80%** | 0.9900 | ~20 dk |
| LightGBM | 87.70% | 82.38% | 98.80% | 89.84% | 0.9869 | **2.6s** |
| XGBoost | 87.40% | 82.00% | 98.82% | 89.62% | 0.9853 | 8.14s |
| **Ensemble** | 95.28% | 92.23% | **99.84%** | 95.88% | **0.9984** | ~2 dk |

---

## 🚀 Hızlı Başlangıç

### 1. Kurulum

```bash
# Repository'yi klonlayın
git clone <repository-url>
cd siber_saldırı_project

# Gerekli paketleri yükleyin
pip install -r requirements.txt
```

### 2. Veri Hazırlama

```bash
# Tüm datasetleri işle
python run_all_preprocessing.py

# Veya tek tek:
python data_preprocessing.py           # KDD Cup 1999
python data_preprocessing_cicids.py    # CICIDS2018
python data_preprocessing_unsw.py      # UNSW-NB15
```

### 3. LSTM Veri Hazırlama

```bash
python prepare_lstm_data_kdd.py
python prepare_lstm_data_cicids.py
python prepare_lstm_data_unsw.py
```

### 4. Model Eğitimi

#### KDD Cup 1999
```bash
python lstm_model.py
python cnn_model.py
python lightgbm_model.py
python xgboost_model.py
python ensemble_model.py
```

#### CICIDS2018
```bash
python lstm_model_cicids.py
python cnn_model_cicids.py
python lightgbm_model_cicids.py
python xgboost_model_cicids.py
python ensemble_model_cicids.py
```

#### UNSW-NB15
```bash
python lstm_model_unsw.py
python cnn_model_unsw.py
python lightgbm_model_unsw.py
python xgboost_model_unsw.py
python ensemble_model_unsw.py
```

### 5. Karşılaştırma ve Raporlama

```bash
# Tüm modelleri karşılaştır
python compare_all_datasets.py

# Sunum için grafikleri görüntüle
```

---

## 📁 Proje Yapısı

```
siber_saldırı_project/
│
├── 📁 datasets/                    # Ham veriler
│   ├── KDD/
│   ├── CICIDS2018/
│   └── UNSW_NB15/
│
├── 📁 processed_data*/             # İşlenmiş veriler (3 klasör)
├── 📁 lstm_data*/                  # LSTM sequences (3 klasör)
│
├── 📁 models/                      # Eğitilmiş modeller (66+ dosya)
│   ├── best_*_model*.keras         # Derin öğrenme modelleri
│   ├── lightgbm_model*.txt         # LightGBM modelleri
│   ├── xgboost_model*.json         # XGBoost modelleri
│   ├── *_confusion_matrix.png      # Confusion matrices
│   ├── *_training_history.png      # Training grafikleri
│   ├── *_feature_importance.png    # Feature importance
│   ├── comprehensive*.png          # Karşılaştırma grafikleri
│   └── *_report.txt                # Performans raporları
│
├── 📄 Veri İşleme
│   ├── data_preprocessing*.py      # Veri ön işleme (3 adet)
│   ├── prepare_lstm_data*.py       # LSTM veri hazırlama (3 adet)
│   └── run_all_preprocessing.py    # Master preprocessing
│
├── 📄 Model Eğitim Scriptleri
│   ├── lstm_model*.py              # LSTM eğitim (3 adet)
│   ├── cnn_model*.py               # CNN eğitim (3 adet)
│   ├── lightgbm_model*.py          # LightGBM eğitim (3 adet)
│   ├── xgboost_model*.py           # XGBoost eğitim (3 adet)
│   └── ensemble_model*.py          # Ensemble (3 adet)
│
├── 📄 Karşılaştırma ve Analiz
│   ├── compare_models.py           # KDD model karşılaştırma
│   ├── compare_cicids_models.py    # CICIDS model karşılaştırma
│   └── compare_all_datasets.py     # 12 model karşılaştırma
│
├── 📚 Dokümantasyon
│   ├── README.md                   # Bu dosya
│   ├── FINAL_PROJECT_REPORT.md     # Detaylı final rapor
│   ├── PROJECT_STATUS.md           # Proje durumu
│   ├── UNSW_RESULTS_SUMMARY.md     # UNSW sonuçları
│   └── IMPLEMENTATION_PLAN.md      # İmplementasyon planı
│
└── 📄 Diğer
    ├── requirements.txt            # Python bağımlılıkları
    ├── .gitignore
    └── presentation.html           # Sunum
```

---

## 🔧 Gereksinimler

```txt
tensorflow>=2.0
lightgbm>=3.0
xgboost>=2.0
scikit-learn>=1.0
pandas>=1.3
numpy>=1.20
matplotlib>=3.3
seaborn>=0.11
```

### Sistem Gereksinimleri
- **Python:** 3.8+
- **RAM:** 16 GB (önerilir)
- **Disk:** ~4 GB boş alan
- **GPU:** Opsiyonel (eğitimi hızlandırır)

---

## 📈 Görselleştirmeler

Proje otomatik olarak şu grafikleri oluşturur:

### Model Bazında
- ✅ Training history (loss, accuracy, precision, recall)
- ✅ Confusion matrix
- ✅ ROC curve
- ✅ Feature importance (LightGBM, XGBoost)

### Karşılaştırma
- ✅ 4 model performans karşılaştırması (her dataset için)
- ✅ 12 model comprehensive karşılaştırma
- ✅ Dataset-model performance heatmap
- ✅ Radar charts

**Toplam: 66+ görselleştirme dosyası**

---

## 💡 Kullanım Senaryoları

### 1. Gerçek Zamanlı İzleme
```python
# LightGBM ile hızlı tespit
import lightgbm as lgb
model = lgb.Booster(model_file='models/lightgbm_model_unsw.txt')
prediction = model.predict(network_traffic)
```

### 2. Yüksek Doğruluk
```python
# CNN ile en yüksek accuracy
from tensorflow import keras
model = keras.models.load_model('models/best_cnn_model_unsw.keras')
prediction = model.predict(network_traffic_sequences)
```

### 3. Maksimum Güvenlik
```python
# Ensemble ile minimum false negative
from ensemble_model_unsw import UNSWEnsembleDetector
detector = UNSWEnsembleDetector()
prediction = detector.predict(X_test)  # 99.84% recall
```

---

## 🎯 Model Seçim Rehberi

| İhtiyaç | Önerilen Model | Sebep |
|---------|---------------|-------|
| **Hız** | LightGBM | Milisaniyeler içinde tahmin |
| **Doğruluk** | CNN | %98.55 accuracy |
| **Güvenlik** | Ensemble | %99.84 recall |
| **Yorumlanabilirlik** | LightGBM/XGBoost | Feature importance |
| **Genel Kullanım** | Ensemble | En dengeli performans |

---

## 📊 Proje İstatistikleri

- ✅ **15 model** eğitildi
- ✅ **3 farklı dataset** analiz edildi
- ✅ **66+ görselleştirme** oluşturuldu
- ✅ **~5 saat** toplam eğitim süresi
- ✅ **~3.5 GB** toplam proje boyutu
- ✅ **40+ Python scripti** geliştirildi

---

## 🏆 Başarılar ve Katkılar

### Akademik Katkılar
- 3 farklı dataset üzerinde kapsamlı karşılaştırma
- Ensemble model performans analizi
- Gerçek zamanlı kullanım önerileri

### Teknik Başarılar
- %98.55 accuracy (UNSW-NB15 CNN)
- %99.84 recall (UNSW-NB15 Ensemble)
- <0.1s tahmin süresi (LightGBM)

---

## 🚀 Gelecek Geliştirmeler

- [ ] Multi-class classification (saldırı türlerini ayırma)
- [ ] Web dashboard (Flask/Streamlit)
- [ ] REST API (model serving)
- [ ] Docker containerization
- [ ] Real-time streaming veri desteği
- [ ] Explainable AI (SHAP, LIME)

---

## 📚 Referanslar

### Veri Setleri
- [KDD Cup 1999](https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
- [CICIDS2018](https://www.unb.ca/cic/datasets/ids-2018.html)
- [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

### Araçlar
- [TensorFlow](https://www.tensorflow.org/)
- [LightGBM](https://lightgbm.readthedocs.io/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [scikit-learn](https://scikit-learn.org/)

---

## 👥 Geliştirici

**Nefise**  
Bilgisayar Mühendisliği 4. Sınıf  
Bitirme Projesi - 2026

---

## 📄 Lisans

Bu proje akademik amaçlar için geliştirilmiştir. Uygun atıf yapılarak kullanılabilir.

---

## 📞 İletişim

Proje hakkında sorularınız için:
- 📧 Email: nefiseturgut60@gmail.com
- 🔗 LinkedIn: Nefise Turgut
- 📁 GitHub: nefiseturgut

---

## ⭐ Teşekkürler

Bu projeyi beğendiyseniz yıldız vermeyi unutmayın! ⭐

---
  
**Versiyon:** 2.1   
**Durum:** ✅ Devam Ediyor
