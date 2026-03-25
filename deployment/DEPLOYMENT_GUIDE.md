# 🚀 Model Deployment Rehberi

**Siber Saldırı Tespit Sistemi - Production Deployment**

---

## 📋 İçindekiler

1. [Genel Bakış](#genel-bakış)
2. [Deployment Seç enekleri](#deployment-seçenekleri)
3. [Hızlı Başlangıç](#hızlı-başlangıç)
4. [Production Deployment](#production-deployment)
5. [Monitoring ve Bakım](#monitoring-ve-bakım)
6. [Sorun Giderme](#sorun-giderme)

---

## 🎯 Genel Bakış

Bu rehber, eğitilmiş modellerinizi production ortamında kullanılabilir hale getirmek için **3 farklı deployment yöntemi** sunar:

### Deployment Mimarisi

```
┌─────────────────────────────────────────────────────────────┐
│                    CLIENT APPLICATIONS                       │
│        (Web Browser, Mobile App, IoT Devices)               │
└──────────────┬──────────────────────┬──────────────────────┘
               │                      │
               ▼                      ▼
┌──────────────────────┐    ┌──────────────────────┐
│  Streamlit Dashboard │    │     REST API         │
│  (Port 8501)         │    │   (Port 5000)        │
└──────────┬───────────┘    └──────────┬───────────┘
           │                           │
           │      ┌────────────────────┘
           │      │
           ▼      ▼
┌─────────────────────────────────────┐
│        Model Server                 │
│  ┌─────────┐  ┌─────────┐          │
│  │  LSTM   │  │   CNN   │          │
│  └─────────┘  └─────────┘          │
│  ┌─────────┐  ┌─────────┐          │
│  │LightGBM │  │ XGBoost │          │
│  └─────────┘  └─────────┘          │
└─────────────────────────────────────┘
```

---

## 🎯 Deployment Seçenekleri

### Seçenek 1: Flask REST API ⚡ (Önerilir)

**Avantajlar:**
- ✅ HTTP üzerinden erişim
- ✅ Çoklu client desteği
- ✅ Production-ready
- ✅ Programatik kullanım

**Kullanım Alanları:**
- Mobil uygulamalar
- Web servisleri
- IoT cihazları
- Otomasyon sistemleri

**Performans:**
- 📊 100+ tahmin/saniye
- ⏱️ <50ms response time
- 💾 ~500MB RAM

---

### Seçenek 2: Docker Container 🐳 (Taşınabilirlik)

**Avantajlar:**
- ✅ Ortamdan bağımsız
- ✅ Kolay deployment
- ✅ Skalabilite
- ✅ CI/CD uyumlu

**Kullanım Alanları:**
- Cloud deployment (AWS, Azure, GCP)
- Kubernetes
- Multi-environment
- Team collaboration

**Performans:**
- 📊 API ile aynı
- 💾 ~1.5GB disk
- 🔄 Saniyeler içinde başlatma

---

### Seçenek 3: Streamlit Dashboard 📊 (Kullanıcı Arayüzü)

**Avantajlar:**
- ✅ Görsel arayüz
- ✅ Real-time monitoring
- ✅ Kolay kullanım
- ✅ Demo için ideal

**Kullanım Alanları:**
- Demo sunumları
- İç kullanım
- Monitoring
- Analiz

**Performans:**
- 👥 10-20 kullanıcı
- ⏱️ Real-time updates
- 💾 ~300MB RAM

---

## 🚀 Hızlı Başlangıç

### 1️⃣ Flask REST API (5 dakika)

```bash
# 1. Bağımlılıkları yükle
cd deployment
pip install -r requirements_deployment.txt

# 2. API'yi başlat
python api_server.py

# 3. Test et
curl http://localhost:5000/health

# 4. Tahmin yap
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.1, 0.2, ..., 0.42],
    "model": "unsw_cnn"
  }'
```

**API Endpoints:**

| Endpoint | Method | Açıklama |
|----------|--------|----------|
| `/` | GET | API bilgisi |
| `/health` | GET | Sağlık kontrolü |
| `/models` | GET | Mevcut modeller |
| `/predict` | POST | Tekil tahmin |
| `/predict/batch` | POST | Toplu tahmin |
| `/stats` | GET | İstatistikler |
| `/docs` | GET | Dokümantasyon |

---

### 2️⃣ Docker Deployment (10 dakika)

```bash
# 1. Docker build
cd deployment
docker build -t cyber-attack-api:latest -f Dockerfile ..

# 2. Container çalıştır
docker run -d \
  --name cyber-api \
  -p 5000:5000 \
  -v $(pwd)/../models:/app/models:ro \
  cyber-attack-api:latest

# 3. Logları kontrol et
docker logs -f cyber-api

# 4. Test et
curl http://localhost:5000/health

# Durdur/Kaldır
docker stop cyber-api
docker rm cyber-api
```

**Docker Compose ile:**

```bash
# Tüm servisleri başlat
docker-compose up -d

# Logları görüntüle
docker-compose logs -f

# Durdur
docker-compose down
```

---

### 3️⃣ Streamlit Dashboard (3 dakika)

```bash
# 1. API'yi başlat (başka terminalde)
python api_server.py

# 2. Dashboard'u başlat
pip install streamlit plotly
streamlit run dashboard.py

# 3. Tarayıcıda aç
# http://localhost:8501
```

**Dashboard Özellikleri:**
- 🎯 Tekil tahmin
- 📊 Toplu analiz
- 📈 Tahmin geçmişi
- 📉 Gerçek zamanlı istatistikler
- 🎨 İnteraktif grafikler

---

## 🏭 Production Deployment

### AWS Deployment

#### EC2 Instance

```bash
# 1. EC2 instance oluştur (t2.medium veya üzeri)

# 2. Bağlan
ssh -i your-key.pem ubuntu@your-ec2-ip

# 3. Kurulum
sudo apt update
sudo apt install -y python3-pip docker.io

# 4. Projeyi kopyala
git clone <your-repo>
cd siber_saldırı_project/deployment

# 5. Docker ile çalıştır
sudo docker-compose up -d

# 6. Nginx konfigürasyonu
sudo apt install -y nginx
# nginx.conf dosyasını yapılandır
```

#### Elastic Beanstalk

```bash
# 1. EB CLI yükle
pip install awsebcli

# 2. Başlat
eb init -p docker cyber-attack-api

# 3. Deploy
eb create cyber-attack-env
eb open
```

---

### Azure Deployment

#### Container Instances

```bash
# 1. Azure CLI
az login

# 2. Resource group
az group create --name CyberAttackRG --location eastus

# 3. Container Registry
az acr create --resource-group CyberAttackRG \
  --name cyberattackregistry --sku Basic

# 4. Push image
docker tag cyber-attack-api:latest \
  cyberattackregistry.azurecr.io/cyber-api:v1
docker push cyberattackregistry.azurecr.io/cyber-api:v1

# 5. Deploy
az container create \
  --resource-group CyberAttackRG \
  --name cyber-api \
  --image cyberattackregistry.azurecr.io/cyber-api:v1 \
  --ports 5000
```

---

### Google Cloud Platform

#### Cloud Run

```bash
# 1. GCloud SDK
gcloud init

# 2. Build
gcloud builds submit --tag gcr.io/PROJECT-ID/cyber-api

# 3. Deploy
gcloud run deploy cyber-api \
  --image gcr.io/PROJECT-ID/cyber-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

## 📊 Monitoring ve Bakım

### Prometheus + Grafana

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'cyber-api'
    static_configs:
      - targets: ['localhost:5000']
    metrics_path: '/metrics'
```

### Logging

```python
# API server'a ekle
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
```

### Health Checks

```bash
# Cron job (her 5 dakikada)
*/5 * * * * curl -f http://localhost:5000/health || systemctl restart cyber-api
```

---

## 🔧 Performance Tuning

### Gunicorn (Production Server)

```bash
# api_server.py yerine:
gunicorn -w 4 -b 0.0.0.0:5000 \
  --timeout 60 \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log \
  api_server:app
```

### Nginx Reverse Proxy

```nginx
# /etc/nginx/sites-available/cyber-api
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

### Load Balancing

```yaml
# docker-compose.yml
services:
  api1:
    image: cyber-attack-api
    ports: ["5001:5000"]
  
  api2:
    image: cyber-attack-api
    ports: ["5002:5000"]
  
  nginx:
    image: nginx
    volumes:
      - ./nginx-lb.conf:/etc/nginx/nginx.conf
    ports: ["80:80"]
```

---

## 🐛 Sorun Giderme

### API başlamıyor

```bash
# Log kontrol
docker logs cyber-api

# Port kontrolü
netstat -tulpn | grep 5000

# Model dosyaları kontrolü
ls -lh ../models/
```

### Yavaş tahminler

- Model cache kullan
- Batch prediction tercih et
- GPU enable et (Keras modelleri için)
- LightGBM kullan (en hızlı)

### Memory issues

```bash
# Container memory limiti
docker run --memory="2g" cyber-attack-api

# Swap kullan
docker run --memory="2g" --memory-swap="4g" cyber-attack-api
```

---

## 📚 İleri Seviye

### Auto-scaling (Kubernetes)

```yaml
# deployment.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: cyber-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: cyber-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### CI/CD Pipeline (GitHub Actions)

```yaml
# .github/workflows/deploy.yml
name: Deploy API
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build Docker image
        run: docker build -t cyber-api .
      - name: Deploy to AWS
        run: |
          # Deployment komutları
```

---

## ✅ Checklist

### Development
- [ ] API server çalışıyor
- [ ] Tüm modeller yüklü
- [ ] Test scriptleri başarılı
- [ ] Dashboard çalışıyor

### Production
- [ ] HTTPS aktif
- [ ] Environment variables güvenli
- [ ] Logging yapılandırıldı
- [ ] Monitoring kuruldu
- [ ] Backup strategy var
- [ ] Auto-scaling yapılandırıldı
- [ ] Load balancing aktif
- [ ] Health checks çalışıyor

---

## 🎓 Sonuç

Bu deployment rehberi ile modellerinizi:
- ✅ Lokal development ortamında test edebilir
- ✅ Docker ile containerize edebilir
- ✅ Cloud platformlarda deploy edebilir
- ✅ Production'da güvenle çalıştırabilirsiniz

**İyi deployment'lar!** 🚀

---

**Hazırlayan:** Nefise  
**Tarih:** Ocak 2026  
**Versiyon:** 1.0
