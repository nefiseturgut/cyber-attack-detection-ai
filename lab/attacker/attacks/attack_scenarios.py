#!/usr/bin/env python3
"""
Otomatik Saldırı + Normal Trafik Üreticisi
Attacker container başladığında otomatik çalışır.
Normal trafik ve saldırı trafiğini karışık üretir.
"""

import random
import subprocess
import time
import os
import socket
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ATTACKER")

TARGET_IP  = os.environ.get("TARGET_IP",  "172.20.0.10")
MONITOR_IP = os.environ.get("MONITOR_IP", "172.20.0.30")
ALL_IPS    = [TARGET_IP, MONITOR_IP]


def run(cmd, timeout=15):
    try:
        subprocess.run(cmd, shell=True, timeout=timeout,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


# ── Normal Trafik ─────────────────────────────────────────────────────────────

def normal_http(ip):
    log.info(f"[NORMAL] HTTP GET -> {ip}")
    run(f"curl -s --max-time 3 http://{ip}/ -o /dev/null")

def normal_ping(ip):
    log.info(f"[NORMAL] PING -> {ip}")
    run(f"ping -c 4 -W 1 {ip}")

def normal_tcp(ip, port=80):
    log.info(f"[NORMAL] TCP connect -> {ip}:{port}")
    try:
        s = socket.create_connection((ip, port), timeout=2)
        s.send(b"GET / HTTP/1.0\r\nHost: target\r\n\r\n")
        s.recv(256)
        s.close()
    except Exception:
        pass


# ── Saldırı Trafiği ───────────────────────────────────────────────────────────

def attack_syn_flood(ip, count=80):
    log.info(f"[ATTACK] SYN Flood -> {ip} ({count} paket)")
    run(f"hping3 -S -p 80 --count {count} {ip}", timeout=30)

def attack_port_scan(ip):
    log.info(f"[ATTACK] Port Scan -> {ip}")
    run(f"nmap -sS -F --min-rate 500 {ip}", timeout=30)

def attack_udp_flood(ip, count=60):
    log.info(f"[ATTACK] UDP Flood -> {ip} ({count} paket)")
    run(f"hping3 --udp -p 53 --count {count} {ip}", timeout=20)

def attack_icmp_flood(ip, count=60):
    log.info(f"[ATTACK] ICMP Flood -> {ip} ({count} paket)")
    run(f"hping3 -1 --count {count} {ip}", timeout=20)

def attack_null_scan(ip):
    log.info(f"[ATTACK] NULL Scan -> {ip}")
    run(f"nmap -sN -F {ip}", timeout=20)


# ── Senaryo Döngüsü ───────────────────────────────────────────────────────────

NORMAL_ACTIONS = [normal_http, normal_ping, normal_tcp]
ATTACK_ACTIONS = [attack_syn_flood, attack_port_scan,
                  attack_udp_flood, attack_icmp_flood, attack_null_scan]


def run_scenario():
    log.info("=" * 50)
    log.info("  OTOMATIK SALDIRI SIMULASYONU BASLADI")
    log.info(f"  Hedefler: {ALL_IPS}")
    log.info("=" * 50)

    cycle = 0
    while True:
        cycle += 1
        ip = random.choice(ALL_IPS)
        log.info(f"--- Dongu {cycle} | Hedef: {ip} ---")

        # Her döngüde 5-7 normal trafik
        for _ in range(random.randint(5, 7)):
            fn = random.choice(NORMAL_ACTIONS)
            fn(ip)
            time.sleep(random.uniform(1.0, 3.0))

        # %40 ihtimalle saldırı
        if random.random() < 0.4:
            fn = random.choice(ATTACK_ACTIONS)
            fn(ip)

        time.sleep(random.uniform(8, 15))


if __name__ == "__main__":
    log.info("Container hazir olana kadar bekleniyor (10s)...")
    time.sleep(10)
    run_scenario()
