# 🐳 Siber Saldırı Tespit API - Docker Image

FROM python:3.10-slim

# Çalışma dizini oluştur
WORKDIR /app

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıklarını kopyala ve yükle
COPY requirements_deployment.txt .
RUN pip install --no-cache-dir -r requirements_deployment.txt

# Uygulama dosyalarını kopyala
COPY deployment/ ./deployment/
COPY models/ ./models/

# Port
EXPOSE 5000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models

# Server'ı başlat
CMD ["python", "deployment/api_server.py"]
