# 🛡️ Siber Saldırı Tespit Sistemi (CNN, LSTM, LightGBM)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

**Kapsamlı bir makine öğrenmesi araştırması: 15 Deneysel Modelden -> Optimize Edilmiş 3 Ana Modele**

---

## 📋 Proje Özeti ve Model Seçim Hikayesi

Bu proje, siber saldırı tespiti alanında kapsamlı bir Ar-Ge çalışması olarak başlamıştır. İlk aşamada **3 farklı veri seti** (KDD Cup 1999, CICIDS2018, UNSW-NB15) üzerinde toplam **15 farklı model ve ensemble (hibrit) mimari** başarıyla eğitilip birbirleriyle kıyaslanmıştır.

Projenin üretim (production) ve canlı test (real-time monitoring) aşamasına geçişinde; sistem karmaşasını azaltmak, spesifik hedeflere (hız, yüksek doğruluk ve zaman serisi analizi) net çözümler sunmak amacıyla **en başarılı 3 modelin süzülmesi (daraltılması)** kararı verilmiştir. Eski deneysel çalışmalar veri ve tecrübe olarak arşivlenirken; projenin ana mimarisi "şampiyon" seçilen **CNN, LSTM ve LightGBM** üzerine inşa edilmiştir.

### 🎯 Ana Hedef
Güncel ve klasik boyutlu veri setleri üzerinde kanıtlanmış (ampirik metodlarla seçilmiş) 3 farklı altyapıyı (Derin Öğrenme ve Karar Ağaçları) gerçek zamanlı tespit mekanizmalarına dönüştürmek ve modern siber güvenlik izleme yazılımları için taban oluşturmak.

---

## 🗂️ Kullanılan Veri Setleri

| Dataset | Özellikler | Kayıt Sayısı | Kullanım Amacı |
|---------|-----------|--------------|----------------|
| **KDD Cup 1999** | 41 | ~500,000 | Klasik benchmark |
| **CICIDS2018** | 80 | ~1,000,000 | Modern saldırılar |
| **UNSW-NB15** | 42 | ~257,000 | Dengeli, gerçekçi |

---

## 🤖 Nihai Seçilen "Şampiyon" Modeller

İlk geliştirme aşamasındaki XGBoost ve karmaşık Ensembling (toplama) testlerinden aldığımız detaylı raporlar ışığında, sistemi ileriye taşımak için alana özel **3 ana model** yetkilendirilmiştir:

### 1. CNN (Convolutional Neural Network) - *Doğruluk Lideri*
- 🧠 Derin öğrenme mimarisi
- 🔍 Pattern (Örüntü) tanıma yeteneği yüksek
- 🎯 Testler sonucunda modern veri setlerinde en yüksek accuracy (doğruluk) oranını sağlayan model olmuştur.

### 2. LSTM (Long Short-Term Memory) - *Analiz Lideri*
- 🧠 Derin öğrenme (RNN varyantı)
- 📊 Sequence-based (Zaman Serisi) yaklaşım
- ⏱️ Ağ trafiğinin zaman içerisindeki anormalliklerini kolayca kavrar, uzun vadeli güvenlik analizi için mükemmeldir.

### 3. LightGBM - *Gerçek Zamanlı (Real-Time) Hız Lideri*
- ⚡ Gradient Boosting ağacı mantığı
- 🚀 Çok hızlı eğitim ve en önemlisi **milisaniyelik tahmin süresi**
- 💾 Hafifliğiyle aktif ağ dinleme (sniffing) aşamasında sistemi yormadan anında tepki verebilecek donanıma sahiptir.

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
├── 📁 eski_surumler_ve_testler/    # İlk araştırmalardaki (XGBoost, Ensemble vb.) deneysel referanslar (ARŞİV)
│
├── 📄 Veri İşleme
│   ├── data_preprocessing*.py      # Temel makine öğrenmesi veri onarımı
│   └── prepare_lstm_data*.py       # Derin öğrenme veri hazırlama
│
├── 📄 Ana Model Eğitim Kodları
│   ├── cnn_model*.py               # Seçili CNN Kodları
│   ├── lstm_model*.py              # Seçili LSTM Kodları
│   └── lightgbm_model*.py          # Seçili LightGBM Kodları
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
