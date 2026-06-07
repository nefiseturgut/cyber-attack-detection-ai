FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    tshark \
    tcpdump \
    net-tools \
    iproute2 \
    iputils-ping \
    && apt-get clean

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/captures /app/logs

ENV PYTHONUNBUFFERED=1

CMD ["python3", "/app/capture.py"]