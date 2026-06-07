#!/usr/bin/env python3
import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

log = logging.getLogger("IDS_FE")

TCP_FLAGS = {"F":0x01,"S":0x02,"R":0x04,"P":0x08,"A":0x10,"U":0x20}

def parse_flags(flag_str):
    result = {k:0 for k in TCP_FLAGS}
    try:
        val = int(flag_str, 16)
        for name, bit in TCP_FLAGS.items():
            result[name] = 1 if (val & bit) else 0
    except:
        pass
    return result

class Flow:
    def __init__(self, key):
        self.key        = key
        self.start_time = None
        self.last_time  = None
        self.fwd_pkts   = []
        self.bwd_pkts   = []
        self.fwd_iat    = []
        self.bwd_iat    = []
        self.all_lens   = []
        self.flags      = defaultdict(int)

    def add(self, length, ts, flags, is_fwd):
        if self.start_time is None:
            self.start_time = ts
        iat = (ts - self.last_time) if self.last_time else 0.0
        self.last_time = ts
        self.all_lens.append(length)
        if is_fwd:
            self.fwd_pkts.append(length)
            self.fwd_iat.append(iat)
        else:
            self.bwd_pkts.append(length)
            self.bwd_iat.append(iat)
        for k,v in flags.items():
            if v: self.flags[k] += 1

    @property
    def duration(self):
        if self.start_time and self.last_time:
            return max(0.0, self.last_time - self.start_time)
        return 0.0

    @staticmethod
    def stats(arr):
        if not arr:
            return dict(mean=0,std=0,min=0,max=0,sum=0)
        a = np.array(arr, dtype=float)
        return dict(mean=float(np.mean(a)),std=float(np.std(a)),
                    min=float(np.min(a)),max=float(np.max(a)),sum=float(np.sum(a)))

class FeatureExtractor:
    def __init__(self):
        self.flows: Dict[tuple, Flow] = {}
        log.info("FeatureExtractor hazir.")

    def _key(self, pkt):
        try:
            proto = 0
            src_ip = dst_ip = None
            sp = dp = 0
            if hasattr(pkt,"ip"):
                src_ip, dst_ip = pkt.ip.src, pkt.ip.dst
            elif hasattr(pkt,"ipv6"):
                src_ip, dst_ip = pkt.ipv6.src, pkt.ipv6.dst
            else:
                return None
            if hasattr(pkt,"tcp"):
                proto = 6; sp = int(pkt.tcp.srcport); dp = int(pkt.tcp.dstport)
            elif hasattr(pkt,"udp"):
                proto = 17; sp = int(pkt.udp.srcport); dp = int(pkt.udp.dstport)
            elif hasattr(pkt,"icmp"):
                proto = 1
            return (src_ip, dst_ip, sp, dp, proto)
        except:
            return None

    def _process(self, pkt):
        key = self._key(pkt)
        if not key: return
        try:
            length = int(pkt.length)
            ts     = float(pkt.sniff_timestamp)
            flags  = {}
            if hasattr(pkt,"tcp") and hasattr(pkt.tcp,"flags"):
                flags = parse_flags(pkt.tcp.flags)
        except:
            return

        rev = (key[1],key[0],key[3],key[2],key[4])
        if key in self.flows:
            fkey, is_fwd = key, True
        elif rev in self.flows:
            fkey, is_fwd = rev, False
        else:
            fkey, is_fwd = key, True
            self.flows[fkey] = Flow(fkey)

        self.flows[fkey].add(length, ts, flags, is_fwd)

    def _to_row(self, flow):
        dur  = flow.duration
        fwd  = Flow.stats(flow.fwd_pkts)
        bwd  = Flow.stats(flow.bwd_pkts)
        pkt  = Flow.stats(flow.all_lens)
        fiat = Flow.stats(flow.fwd_iat)
        biat = Flow.stats(flow.bwd_iat)
        aiat = Flow.stats(flow.fwd_iat + flow.bwd_iat)
        fc   = flow.flags
        tp   = len(flow.fwd_pkts) + len(flow.bwd_pkts)
        tb   = fwd["sum"] + bwd["sum"]
        return {
            "src_ip":          flow.key[0],
            "dst_ip":          flow.key[1],
            "src_port":        flow.key[2],
            "dst_port":        flow.key[3],
            "protocol":        flow.key[4],
            "flow_duration":   dur,
            "flow_bytes_s":    tb/dur if dur>0 else 0,
            "flow_pkts_s":     tp/dur if dur>0 else 0,
            "fwd_pkt_count":   len(flow.fwd_pkts),
            "fwd_len_mean":    fwd["mean"],
            "fwd_len_std":     fwd["std"],
            "fwd_len_min":     fwd["min"],
            "fwd_len_max":     fwd["max"],
            "fwd_bytes":       fwd["sum"],
            "fwd_iat_mean":    fiat["mean"],
            "fwd_iat_std":     fiat["std"],
            "bwd_pkt_count":   len(flow.bwd_pkts),
            "bwd_len_mean":    bwd["mean"],
            "bwd_len_std":     bwd["std"],
            "bwd_len_min":     bwd["min"],
            "bwd_len_max":     bwd["max"],
            "bwd_bytes":       bwd["sum"],
            "bwd_iat_mean":    biat["mean"],
            "bwd_iat_std":     biat["std"],
            "pkt_len_mean":    pkt["mean"],
            "pkt_len_std":     pkt["std"],
            "pkt_len_min":     pkt["min"],
            "pkt_len_max":     pkt["max"],
            "iat_mean":        aiat["mean"],
            "iat_std":         aiat["std"],
            "syn_cnt":         fc.get("S",0),
            "fin_cnt":         fc.get("F",0),
            "rst_cnt":         fc.get("R",0),
            "psh_cnt":         fc.get("P",0),
            "ack_cnt":         fc.get("A",0),
            "urg_cnt":         fc.get("U",0),
            "down_up_ratio":   len(flow.bwd_pkts)/len(flow.fwd_pkts) if flow.fwd_pkts else 0,
            "total_pkts":      tp,
            "total_bytes":     tb,
            "timestamp":       datetime.fromtimestamp(flow.start_time).strftime("%Y-%m-%d %H:%M:%S") if flow.start_time else "",
        }

    def extract_from_packets(self, packets) -> Optional[pd.DataFrame]:
        for pkt in packets:
            try: self._process(pkt)
            except: pass
        if not self.flows:
            return None
        rows = []
        done = []
        for key, flow in self.flows.items():
            if len(flow.fwd_pkts)+len(flow.bwd_pkts) < 2:
                continue
            try:
                rows.append(self._to_row(flow))
                done.append(key)
            except Exception as e:
                log.error(f"Row hatasi: {e}")
        for k in done:
            del self.flows[k]
        if not rows:
            return None
        df = pd.DataFrame(rows)
        log.info(f"DataFrame: {len(df)} satir x {len(df.columns)} sutun")
        return df

    def extract_from_pcap(self, path, fmt="cicids") -> Optional[pd.DataFrame]:
        import pyshark
        cap = pyshark.FileCapture(path, use_json=True)
        pkts = list(cap); cap.close()
        return self.extract_from_packets(pkts)

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) > 1:
        fe = FeatureExtractor()
        df = fe.extract_from_pcap(sys.argv[1])
        if df is not None:
            print(f"Sutunlar: {list(df.columns)}")
            print(df.head(3).to_string())
            out = sys.argv[1].replace(".pcap","_features.csv")
            df.to_csv(out, index=False)
            print(f"Kaydedildi: {out}")
    else:
        print("Kullanim: python3 feature_extractor.py dosya.pcap")