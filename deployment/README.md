# 🚀 Deployment - Hızlı Başlangıç

Modellerinizi 5 dakikada çalıştırın!

---

## 📦 Oluşturulan Dosyalar

```
deployment/
├── api_server.py              # Flask REST API server
├── dashboard.py               # Streamlit dashboard
├── test_api.py                # API test scripti
├── Dockerfile                 # Docker image
├── docker-compose.yml         # Docker orchestration
├── requirements_deployment.txt # Python bağımlılıkları
├── DEPLOYMENT_GUIDE.md        # Detaylı rehber
└── README.md                  # Bu dosya
```

---

## ⚡ En Hızlı Yol (3 adım)

### 1. Bağımlılıkları Yükle
```bash
cd deployment
pip install -r requirements_deployment.txt
```

### 2. API'yi Başlat
```bash
python api_server.py
```

### 3. Dashboard'u Başlat (Yeni terminal)
```bash
streamlit run dashboard.py
```

🎉 **Hazır!** Tarayıcınızda `http://localhost:8501` adresine gidin.

---

## 🎯 Kullanım Seçenekleri

### Seçenek 1: REST API Kullanımı

```python
import requests
import numpy as np

# Tahmin yap
features = np.random.randn(42).tolist()

response = requests.post('http://localhost:5000/predict', json={
    'features': features,
    'model': 'unsw_cnn'  # En yüksek accuracy
})

result = response.json()
print(f"Tahmin: {result['result']['prediction']}")
print(f"Güven: {result['result']['confidence']*100:.2f}%")
```

### Seçenek 2: Dashboard Kullanımı

1. Tarayıcıda `http://localhost:8501` açın
2. Model seçin (örn: unsw_cnn)
3. "Rastgele Veri" veya "Örnek Saldırı" seçin
4. "Analiz Et" butonuna tıklayın
5. Sonuçları görüntüleyin!

### Seçenek 3: Curl ile Test

```bash
# Health check
curl http://localhost:5000/health

# Modelleri listele
curl http://localhost:5000/models

# Tahmin yap
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.1, 0.2, 0.3, ..., 0.42],
    "model": "unsw_lightgbm"
  }'
```

---

## 🐳 Docker ile Çalıştırma

```bash
# Build
docker build -t cyber-api -f Dockerfile ..

# Run
docker run -d -p 5000:5000 \
  -v $(pwd)/../models:/app/models:ro \
  cyber-api

# Test
curl http://localhost:5000/health
```

---

## 🧪 Test Etme

```bash
# Otomatik test suite
python test_api.py

# Çıktı:
# ✅ Health Check - PASSED
# ✅ Models Endpoint - PASSED
# ✅ Single Prediction - PASSED
# ✅ Batch Prediction - PASSED
# ✅ Stats Endpoint - PASSED
# ✅ Performance - PASSED
# ✅ Error Handling - PASSED
```

---

## 📊 API Endpoints

| Endpoint | Method | Açıklama | Örnek |
|----------|--------|----------|-------|
| `/health` | GET | Sağlık kontrolü | `curl /health` |
| `/models` | GET | Mevcut modeller | `curl /models` |
| `/predict` | POST | Tekil tahmin | JSON body |
| `/predict/batch` | POST | Toplu tahmin | JSON array |
| `/stats` | GET | İstatistikler | `curl /stats` |
| `/docs` | GET | Dokümantasyon | Tarayıcıda aç |

---

## 🎨 Dashboard Özellikleri

- **Tekil Tahmin**: Tek bir ağ trafiği analizi
- **Toplu Tahmin**: CSV dosyası veya toplu veri analizi
- **Tahmin Geçmişi**: Tüm tahminlerin kaydı
- **Grafikler**: İnteraktif görselleştirmeler
- **Model Karşılaştırma**: Farklı modelleri test et

---

## 🚀 Hangi Model?

| İhtiyaç | Model | Sebep |
|---------|-------|-------|
| **En Yüksek Doğruluk** | `unsw_cnn` | %98.55 accuracy |
| **En Hızlı Tahmin** | `unsw_lightgbm` | 2.6s eğitim, <0.1s tahmin |
| **Dengeli Performans** | `unsw_lstm` | %96.65 accuracy, sequence patterns |
| **Production** | `unsw_lightgbm` | Hız + düşük kaynak |

---

## 💡 İpuçları

### Performans Optimizasyonu
```bash
# Birden fazla worker (production)
gunicorn -w 4 -b 0.0.0.0:5000 api_server:app

# Nginx reverse proxy
# Detaylar: DEPLOYMENT_GUIDE.md
```

### GPU Kullanımı
```python
# api_server.py içinde
import tensorflow as tf

# GPU'yu etkinleştir
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ {len(gpus)} GPU bulundu")
```

### Monitoring
```bash
# API istatistikleri
curl http://localhost:5000/stats

# Çıktı:
# {
#   "total_predictions": 1523,
#   "average_prediction_time_ms": 45.2,
#   "models_loaded": 4
# }
```

---

## 🐛 Sorun Giderme

### API başlamıyor
```bash
# Port kullanımda mı?
netstat -ano | findstr :5000

# Model dosyaları var mı?
dir ..\models\*unsw*
```

### Import hataları
```bash
# Tüm bağımlılıkları tekrar yükle
pip install -r requirements_deployment.txt --force-reinstall
```

### Model bulunamıyor
```bash
# Model yolunu kontrol et
# api_server.py içinde path'leri düzenleyin:
# '../models/best_lstm_model_unsw.keras'
```

---

## 📚 Daha Fazla Bilgi

- **Detaylı Rehber**: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **API Dokümantasyonu**: `http://localhost:5000/docs`
- **Cloud Deployment**: DEPLOYMENT_GUIDE.md → Production Deployment
- **CI/CD**: DEPLOYMENT_GUIDE.md → Advanced

---

## ✅ Checklist

Deployment öncesi kontrol listesi:

- [ ] API başlatıldı (`http://localhost:5000/health` → status: healthy)
- [ ] Tüm modeller yüklü (4 model görünüyor)
- [ ] Test scripti başarılı (7/7 test passed)
- [ ] Dashboard açılıyor (`http://localhost:8501`)
- [ ] Tahmin çalışıyor (örnek tahmin başarılı)

---

## 🎓 Özet

**3 Deployment Yöntemi:**

1. **REST API** → Programatik kullanım, production
2. **Docker** → Taşınabilirlik, cloud deployment
3. **Dashboard** → Görsel arayüz, demo, analiz

**Hepsi hazır ve kullanıma sunuldu!** 🎉

---

**Hazırlayan:** Nefise  
**Tarih:** Ocak 2026  
**Durum:** ✅ Production Ready
