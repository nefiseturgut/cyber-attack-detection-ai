# 🎯 ÇOK-DATASET ENSEMBLE PROJESİ - İMPLEMENTASYON PLANI

## 📋 Proje Özeti

Bu proje, **3 farklı siber saldırı detection dataset'i** kullanarak **kapsamlı bir ensemble model** oluşturmayı amaçlamaktadır:

1. **KDD Cup 1999** - Klasik benchmark dataset (41 özellik)
2. **CICIDS2018** - Modern güncel dataset (80 özellik)  
3. **UNSW_NB15** - Balanced dataset (45 özellik)

Her dataset için **4 farklı model** eğitilecek:
- LSTM (Deep Learning)
- CNN (Deep Learning)
- LightGBM (Gradient Boosting)
- XGBoost (Gradient Boosting)

**Toplam: 12 model + Ensemble**



## 🗂️ Yeni Proje Yapısı

```
siber_saldırı_project/
│
├── 📁 datasets/
│   ├── KDD/
│   │   ├── KDDTrain+.txt
│   │   └── KDDTest+.txt
│   ├── CICIDS2018/
│   │   └── Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv
│   └── UNSW_NB15/
│       ├── UNSW_NB15_training-set.csv
│       └── UNSW_NB15_testing-set.csv
│
├── 📁 processed_data_kdd/         ← KDD işlenmiş veri
│   ├── X_train.npy, y_train.npy
│   ├── X_test.npy, y_test.npy
│   └── feature_names.txt
│
├── 📁 processed_data_cicids/      ← CICIDS işlenmiş veri
│   ├── X_train.npy, y_train.npy
│   ├── X_test.npy, y_test.npy
│   └── feature_names.txt
│
├── 📁 processed_data_unsw/        ← UNSW işlenmiş veri
│   ├── X_train.npy, y_train.npy
│   ├── X_test.npy, y_test.npy
│   └── feature_names.txt
│
├── 📁 lstm_data_kdd/              ← KDD LSTM sequences
├── 📁 lstm_data_cicids/           ← CICIDS LSTM sequences
├── 📁 lstm_data_unsw/             ← UNSW LSTM sequences
│
├── 📁 models_kdd/                 ← KDD modelleri
│   ├── lstm_model.keras
│   ├── cnn_model.keras
│   ├── lightgbm_model.txt
│   ├── xgboost_model.json
│   └── *.png (grafikler)
│
├── 📁 models_cicids/              ← CICIDS modelleri
│   ├── lstm_model.keras
│   ├── cnn_model.keras
│   ├── lightgbm_model.txt
│   ├── xgboost_model.json
│   └── *.png (grafikler)
│
├── 📁 models_unsw/                ← UNSW modelleri
│   ├── lstm_model.keras
│   ├── cnn_model.keras
│   ├── lightgbm_model.txt
│   ├── xgboost_model.json
│   └── *.png (grafikler)
│
├── 📁 ensemble_models/            ← Ensemble sonuçları
│   ├── ensemble_weights.npy
│   ├── ensemble_predictions.npy
│   └── ensemble_comparison.png
│
├── 📁 final_results/              ← Final karşılaştırma
│   ├── 12_model_comparison.png
│   ├── dataset_comparison.png
│   ├── final_report.pdf
│   └── final_report.md
│
└── 📄 Python Scriptleri
    ├── data_preprocessing.py          (KDD)
    ├── data_preprocessing_cicids.py   (CICIDS)
    ├── data_preprocessing_unsw.py     (UNSW)
    ├── run_all_preprocessing.py       (Master preprocessing)
    │
    ├── train_models_kdd.py            (KDD modelleri eğit)
    ├── train_models_cicids.py         (CICIDS modelleri eğit)
    ├── train_models_unsw.py           (UNSW modelleri eğit)
    ├── run_all_training.py            (Master training)
    │
    ├── create_ensemble.py             (Ensemble oluştur)
    ├── compare_all_datasets.py        (Dataset karşılaştırma)
    ├── generate_final_report.py       (Final rapor)
    │
    └── README_ENSEMBLE.md             (Yeni dokümantasyon)
```

---

## 🚀 İMPLEMENTASYON ADIMLARI

### ✅ AŞAMA 1: Veri Ön İşleme (2 saat)

**Durum:** %66 Tamamlandı

- [x] `data_preprocessing.py` - KDD Cup 1999 ✅
- [x] `data_preprocessing_cicids.py` - CICIDS2018 ✅
- [x] `data_preprocessing_unsw.py` - UNSW_NB15 ✅
- [ ] `run_all_preprocessing.py` - Master script (Oluşturuldu, test edilecek)

**Çıktı:**
```
processed_data_kdd/      (41 özellik)
processed_data_cicids/   (80 özellik)
processed_data_unsw/     (45 özellik)
```

**Komut:**
```bash
python run_all_preprocessing.py
```

---

### 🔄 AŞAMA 2: LSTM Veri Hazırlama (1 saat)

**Durum:** %0 Tamamlandı

- [ ] `prepare_lstm_data_kdd.py` - KDD sequences
- [ ] `prepare_lstm_data_cicids.py` - CICIDS sequences
- [ ] `prepare_lstm_data_unsw.py` - UNSW sequences

**Çıktı:**
```
lstm_data_kdd/      (sequence_length=10)
lstm_data_cicids/   (sequence_length=10)
lstm_data_unsw/     (sequence_length=10)
```

---

### 🧠 AŞAMA 3: Model Eğitimi (4-6 saat)

**Durum:** %0 Tamamlandı

#### 3.1: KDD Cup 1999 Modelleri
- [ ] LSTM model (lstm_model_kdd.py)
- [ ] CNN model (cnn_model_kdd.py)
- [ ] LightGBM model (lightgbm_model_kdd.py)
- [ ] XGBoost model (xgboost_model_kdd.py)

#### 3.2: CICIDS2018 Modelleri
- [ ] LSTM model (lstm_model_cicids.py)
- [ ] CNN model (cnn_model_cicids.py)
- [ ] LightGBM model (lightgbm_model_cicids.py)
- [ ] XGBoost model (xgboost_model_cicids.py)

#### 3.3: UNSW_NB15 Modelleri
- [ ] LSTM model (lstm_model_unsw.py)
- [ ] CNN model (cnn_model_unsw.py)
- [ ] LightGBM model (lightgbm_model_unsw.py)
- [ ] XGBoost model (xgboost_model_unsw.py)

**Toplam: 12 model**

---

### 🎭 AŞAMA 4: Ensemble Oluşturma (1 saat)

**Durum:** %0 Tamamlandı

- [ ] `create_ensemble.py` - Voting ensemble
- [ ] `create_stacked_ensemble.py` - Stacking ensemble
- [ ] `ensemble_comparison.py` - Ensemble vs Individual

**Ensemble Stratejileri:**
1. **Hard Voting** - Çoğunluk oyu
2. **Soft Voting** - Probability ortalaması
3. **Weighted Voting** - Performansa göre ağırlıklı
4. **Stacking** - Meta-classifier

---

### 📊 AŞAMA 5: Karşılaştırma ve Raporlama (2 saat)

**Durum:** %0 Tamamlandı

#### 5.1: Dataset Karşılaştırması
- [ ] `compare_datasets.py`
- [ ] Her dataset için en iyi model
- [ ] Dataset zorluğu analizi

#### 5.2: Model Karşılaştırması
- [ ] `compare_all_models.py`
- [ ] 12 modelin performans matrisi
- [ ] Model-Dataset uyumu analizi

#### 5.3: Görselleştirmeler
- [ ] 12 modeli içeren comprehensive comparison chart
- [ ] Dataset comparison heatmap
- [ ] Ensemble improvement chart
- [ ] ROC curves (12 model)
- [ ] Confusion matrices (12 model)

#### 5.4: Final Rapor
- [ ] `generate_final_report.py`
- [ ] Markdown rapor
- [ ] PDF rapor
- [ ] README güncelleme

---

## 📈 BEKLENEN ÇIKTILAR

### 1. Performans Matrisi (Örnek)

```
┌──────────────┬─────────┬─────────┬──────────┬──────────┐
│ Model/Dataset│   KDD   │ CICIDS  │  UNSW    │ Ensemble │
├──────────────┼─────────┼─────────┼──────────┼──────────┤
│ LSTM         │ 99.2%   │ 97.8%   │ 94.5%    │          │
│ CNN          │ 98.7%   │ 96.9%   │ 93.2%    │          │
│ LightGBM     │ 98.5%   │ 98.1%   │ 95.1%    │          │
│ XGBoost      │ 98.9%   │ 98.3%   │ 94.8%    │          │
├──────────────┼─────────┼─────────┼──────────┼──────────┤
│ Best/Dataset │ 99.2%   │ 98.3%   │ 95.1%    │          │
│ ENSEMBLE-ALL │         │         │          │ 99.7% ⭐  │
└──────────────┴─────────┴─────────┴──────────┴──────────┘
```

### 2. Dataset Analizi

- **KDD Cup 1999**: En yüksek accuracy, eski dataset
- **CICIDS2018**: Modern saldırılar, zorlu
- **UNSW_NB15**: Balanced, en gerçekçi
- **Ensemble**: Tüm datasetlerin güçlü yönlerini birleştirir

### 3. Model Insights

- Hangi model hangi dataset'te daha iyi?
- Hangi saldırı türleri zor tespit ediliyor?
- Feature importance karşılaştırması
- Eğitim süresi vs performans

---

## ⏱️ ZAMAN ÇİZELGESİ

| Aşama | Süre | Açıklama |
|-------|------|----------|
| Preprocessing | 2 saat | 3 dataset işleme |
| LSTM Data Prep | 1 saat | Sequence oluşturma |
| Model Training | 4-6 saat | 12 model eğitimi |
| Ensemble | 1 saat | Birleştirme ve test |
| Reporting | 2 saat | Görselleştirme ve rapor |
| **TOPLAM** | **10-12 saat** | Tüm proje |

---

## 🎯 SONRAKI ADIMLAR

### ŞİMDİ YAPILACAKLAR:

1. ✅ Preprocessing scriptleri tamamlandı
2. ⏳ CICIDS preprocessing çalışıyor (büyük dosya)
3. ✅ UNSW preprocessing tamamlandı
4. 🔜 Tüm preprocessing'lerin bitmesini bekle
5. 🔜 LSTM veri hazırlama scriptleri oluştur
6. 🔜 Model eğitim scriptleri adapte et

### KULLANICI ONAY BEKLENİYOR:

- [ ] Preprocessing tamamlandıktan sonra model eğitimine geçilsin mi?
- [ ] Her aşama otomatik çalışsın mı yoksa manuel onay mı?
- [ ] GPU kullanılabilir mi? (Eğitim hızlandırma için)

---

## 💡 NOTLAR

1. **Veri Boyutu**: CICIDS çok büyük (~1M satır), subsampling gerekebilir
2. **Eğitim Süresi**: GPU kullanılırsa 50% daha hızlı
3. **Disk Alanı**: Toplam ~2-3 GB model dosyası
4. **RAM**: En az 16 GB önerilir (CICIDS için)

---

**Proje Durumu:** 🟡 İlerleme Aşamasında  
**Tamamlanma:** %15  
**Tahmini Bitiş:** 10-12 saat  
**Son Güncelleme:** 5 Ocak 2026
