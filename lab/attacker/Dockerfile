FROM kalilinux/kali-rolling:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    nmap \
    hping3 \
    hydra \
    net-tools \
    iputils-ping \
    iproute2 \
    python3 \
    python3-pip \
    python3-scapy \
    && apt-get clean

WORKDIR /opt/attacks
CMD ["tail", "-f", "/dev/null"]