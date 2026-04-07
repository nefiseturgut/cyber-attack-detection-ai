# 🛡️ Siber Saldırı Tespit Sistemi (CNN, LSTM, LightGBM)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

**Kapsamlı bir makine öğrenmesi projesi: 3 Dataset × 3 Temel Model = 9 Eğitilmiş Model**

---

## 📋 Proje Özeti

Bu proje, **3 farklı siber saldırı tespit veri seti** kullanarak siber dünya dinamiklerine en uygun olan **3 farklı makine öğrenmesi modeline (CNN, LSTM, LightGBM)** odaklanmış ve yüksek performanslı siber saldırı tespit sistemleri oluşturmuştur. Karmaşayı azaltmak ve optimum sonuçlar elde etmek adına en verimli teknolojiler seçilmiştir.

### 🎯 Ana Hedef
Güncel ve klasik boyutlu veri setleri üzerinde, farklı altyapılara (Derin Öğrenme ve Karar Ağaçları) sahip 3 ana modeli karşılaştırmak, gerçek zamanlı tespit mekanizmaları geliştirmek ve modern siber güvenlik yazılımları için taban oluşturmak.

---

## 🗂️ Kullanılan Veri Setleri

| Dataset | Özellikler | Kayıt Sayısı | Kullanım Amacı |
|---------|-----------|--------------|----------------|
| **KDD Cup 1999** | 41 | ~500,000 | Klasik benchmark |
| **CICIDS2018** | 80 | ~1,000,000 | Modern saldırılar |
| **UNSW-NB15** | 42 | ~257,000 | Dengeli, gerçekçi |

---

## 🤖 Geliştirilen Modeller

Her veri seti için performansları optimize edilmiş **3 farklı model türü** geliştirilmiştir:

### 1. CNN (Convolutional Neural Network) - *Doğruluk Lideri*
- 🧠 Derin öğrenme mimarisi
- 🔍 Pattern (Örüntü) tanıma yeteneği yüksek
- 🎯 En yüksek accuracy (doğruluk) oranını sağlar

### 2. LSTM (Long Short-Term Memory) - *Analiz Lideri*
- 🧠 Derin öğrenme (RNN varyantı)
- 📊 Sequence-based (Zaman Serisi) yaklaşım
- ⏱️ Ağ trafiğinin zaman içerisindeki anormalliklerini kolayca kavrar

### 3. LightGBM - *Gerçek Zamanlı (Real-Time) Hız Lideri*
- ⚡ Gradient Boosting ağacı mantığı
- 🚀 Çok hızlı eğitim ve tahmin süresi (milisaniyeler)
- 💾 Hafif, düşük donanım kaynağı tüketir ve production ortamları için optimaldir.

---

## 📊 Performans Sonuçları (Özet)

### 🏆 En İyi Sonuçlar

| Dataset | En İyi Model | Öne Çıkan Başarı | Sebep / Özellik |
|---------|-------------|-------------------|-----------------|
| **KDD Cup 1999** | LightGBM | **80.21%** Accuracy | Saniyeler içinde en yüksek isabet |
| **CICIDS2018** | CNN | **96.00%** Accuracy | Modern saldırı tespitinde lider |
| **UNSW-NB15** | CNN | **98.55%** Accuracy | Güçlü pattern yakalama becerisi |

### 📈 UNSW-NB15 Detaylı Performans (Örnekleme)

| Model | Accuracy | Precision | Recall | F1-Score | AUC | Eğitim Süresi |
|-------|----------|-----------|--------|----------|-----|---------------|
| LSTM | 96.65% | 95.00% | 97.00% | 96.00% | 0.9800 | ~4 dk |
| **CNN** | **98.55%** | 97.00% | 98.64% | **97.80%** | 0.9900 | ~20 dk |
| LightGBM | 87.70% | 82.38% | 98.80% | 89.84% | 0.9869 | **2.6s** |

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

# LSTM için sekans (sequence) verilerini hazırla
python prepare_lstm_data_kdd.py
python prepare_lstm_data_cicids.py
python prepare_lstm_data_unsw.py
```

### 3. Model Eğitimi

İstediğiniz veri seti için ilgili modeli direkt çalıştırabilirsiniz. Örnek (UNSW-NB15):

```bash
python cnn_model_unsw.py
python lstm_model_unsw.py
python lightgbm_model_unsw.py
```

---

## 📁 Proje Yapısı

```text
siber_saldırı_project/
│
├── 📁 datasets/                    # Ham veriler
├── 📁 processed_data*/             # İşlenmiş veriler
├── 📁 lstm_data*/                  # LSTM formatına sokulmuş matrisler
│
├── 📁 eski_surumler_ve_testler/    # Projenin geçmişte test edilmiş diğer algoritmaları (XGBoost, Ensemble vb.)
│
├── 📄 Veri İşleme
│   ├── data_preprocessing*.py      # Temel makine öğrenmesi veri onarımı
│   └── prepare_lstm_data*.py       # Derin öğrenme veri hazırlama
│
├── 📄 Ana Model Eğitim Kodları
│   ├── cnn_model*.py               # CNN Training
│   ├── lstm_model*.py              # LSTM Training
│   └── lightgbm_model*.py          # LightGBM Training
│
├── 📚 Dokümantasyon
│   ├── README.md                   # Bu dosya
│   └── ROADMAP.md                  # İleriye dönük monitoring planları
│
└── 📄 Diğer (Gereksinimler vs.)
```

---

## 🔧 Gereksinimler

- **Python:** 3.8+
- tensorflow>=2.0
- lightgbm>=3.0
- scikit-learn>=1.0
- pandas>=1.3, numpy>=1.20
- matplotlib, seaborn

*(Detaylar için `requirements.txt` dosyasına bakabilirsiniz)*

---

## 💡 Hangi Modeli Kullanmalıyım? (Model Seçim Rehberi)

| İhtiyaç | Önerilen Model | Neden? |
|---------|---------------|-------|
| **Canlı (Gerçek Zamanlı) Tespit** | **LightGBM** | Milisaniyeler içinde reaksiyon verebilir, sistem kaynağı yormaz. |
| **Maksimum Doğruluk (Accuracy)** | **CNN** | %98.55'e varan oranlarla pattern çıkarımında hata yapma payı en düşüktür. |
| **Zaman / Ağ Akışı Analizi** | **LSTM** | Paketlerin kronolojik akışını hesaba kattığı için karmaşık DDoS/zaman serisi ataklarını daha iyi geneller. |

---

## 👥 Geliştirici

**Nefise**  
Bilgisayar Mühendisliği 4. Sınıf  
Bitirme Projesi - 2026

## 📄 Lisans
Bu proje akademik amaçlar için geliştirilmiştir. Uygun atıf yapılarak kullanılabilir.
