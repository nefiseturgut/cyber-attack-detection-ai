#!/usr/bin/env python3
"""
IDS Streamlit Dashboard
Canlı trafik izleme, saldırı alarmları ve model karşılaştırma paneli.
Çalıştırma: streamlit run app.py
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Sayfa ayarları ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IDS Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ALERTS_FILE  = Path(os.environ.get("ALERTS_FILE", "/app/logs/alerts.jsonl"))
REFRESH_SEC  = int(os.environ.get("REFRESH_SEC", "3"))
MAX_ROWS     = 500   # bellekte tutulacak maksimum kayıt

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.alarm-box {
    background: #ff4b4b22;
    border: 2px solid #ff4b4b;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
}
.safe-box {
    background: #00c85322;
    border: 2px solid #00c853;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
}
.metric-label { font-size: 13px; color: #888; }
</style>
""", unsafe_allow_html=True)

# ── Veri yükleme ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=REFRESH_SEC)
def load_alerts(path: str, max_rows: int) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows[-max_rows:])
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    return df

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/IDS-Monitor-blue?style=for-the-badge", use_container_width=True)
    st.title("⚙️ Ayarlar")

    refresh = st.slider("Yenileme aralığı (sn)", 1, 30, REFRESH_SEC)
    time_window = st.selectbox("Zaman penceresi", ["Son 5 dk", "Son 15 dk", "Son 1 saat", "Tümü"], index=0)
    show_only_attacks = st.toggle("Yalnızca saldırıları göster", value=False)

    st.divider()
    st.caption(f"📁 Log: `{ALERTS_FILE}`")
    st.caption(f"🔄 Her {refresh}s yenileniyor")

# ── Başlık ────────────────────────────────────────────────────────────────────
st.title("🛡️ Siber Saldırı Tespit Sistemi — Canlı İzleme")
st.caption(f"Son güncelleme: {datetime.now().strftime('%H:%M:%S')}")

# ── Veri yükle ────────────────────────────────────────────────────────────────
df_all = load_alerts(str(ALERTS_FILE), MAX_ROWS)

if df_all.empty:
    st.info("📡 Henüz trafik verisi yok. Monitor container çalışıyor mu?")
    st.stop()

# Zaman filtresi
now = pd.Timestamp.now(tz=None)
windows = {
    "Son 5 dk":   timedelta(minutes=5),
    "Son 15 dk":  timedelta(minutes=15),
    "Son 1 saat": timedelta(hours=1),
    "Tümü":       None,
}
w = windows[time_window]
if w:
    df = df_all[df_all["ts"] >= (now - w)].copy()
else:
    df = df_all.copy()

if show_only_attacks:
    df = df[df["is_attack"]]

if df.empty:
    st.warning("Bu zaman penceresinde kayıt yok.")
    st.stop()

total      = len(df)
n_attack   = int(df["is_attack"].sum())
n_normal   = total - n_attack
attack_pct = n_attack / total * 100 if total else 0

# ── Alarm kutusu ──────────────────────────────────────────────────────────────
if n_attack > 0:
    last_atk = df[df["is_attack"]].iloc[-1]
    st.markdown(f"""
    <div class="alarm-box">
    <h3>🚨 SALDIRI TESPİT EDİLDİ</h3>
    <b>Son saldırı:</b> {last_atk.get('src_ip','?')} → {last_atk.get('dst_ip','?')}
    &nbsp;|&nbsp; Port: {int(last_atk.get('dst_port',0))}
    &nbsp;|&nbsp; {str(last_atk.get('ts',''))[:19]}
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown('<div class="safe-box"><h3>✅ Trafik Normal</h3></div>', unsafe_allow_html=True)

# ── KPI metrikleri ────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Toplam Akış",  f"{total:,}")
c2.metric("Saldırı",      f"{n_attack:,}", delta=f"{attack_pct:.1f}%",
          delta_color="inverse" if n_attack else "off")
c3.metric("Normal",       f"{n_normal:,}")
c4.metric("Ort. Gecikme (ms)",
          f"{df['latency_ms'].mean():.1f}" if "latency_ms" in df.columns else "—")

st.divider()

# ── Grafikler ─────────────────────────────────────────────────────────────────
col_pie, col_line = st.columns([1, 2])

with col_pie:
    st.subheader("Trafik Dağılımı")
    pie_data = pd.DataFrame({
        "Tür":   ["Normal", "Saldırı"],
        "Sayı":  [n_normal, n_attack],
    })
    st.bar_chart(pie_data.set_index("Tür"), color=["#00c853", "#ff4b4b"])

with col_line:
    st.subheader("Akış Zaman Serisi")
    ts_df = df.set_index("ts").resample("30s").agg(
        Toplam=("is_attack", "count"),
        Saldırı=("is_attack", "sum"),
    ).fillna(0)
    ts_df["Normal"] = ts_df["Toplam"] - ts_df["Saldırı"]
    st.line_chart(ts_df[["Normal", "Saldırı"]])

st.divider()

# ── Model karşılaştırma ───────────────────────────────────────────────────────
if all(c in df.columns for c in ["lgbm", "cnn", "lstm"]):
    st.subheader("🤖 Model Tahminleri Karşılaştırması")
    mc1, mc2, mc3 = st.columns(3)
    for col_w, model_col, label in [
        (mc1, "lgbm", "⚡ LightGBM"),
        (mc2, "cnn",  "🧠 CNN"),
        (mc3, "lstm", "📊 LSTM"),
    ]:
        n = int(df[model_col].sum())
        col_w.metric(f"{label} — Saldırı", f"{n:,}",
                     delta=f"{n/total*100:.1f}%" if total else "0%",
                     delta_color="inverse" if n else "off")

    st.divider()

# ── Canlı log tablosu ─────────────────────────────────────────────────────────
st.subheader("📋 Son Akışlar")
display_cols = [c for c in ["ts", "src_ip", "dst_ip", "src_port", "dst_port",
                             "is_attack", "lgbm", "cnn", "lstm", "mode", "latency_ms"]
                if c in df.columns]
log_df = df[display_cols].tail(50).sort_values("ts", ascending=False)

def highlight_attacks(row):
    color = "#ff4b4b33" if row.get("is_attack") else ""
    return [f"background-color: {color}"] * len(row)

st.dataframe(
    log_df.style.apply(highlight_attacks, axis=1),
    use_container_width=True,
    height=400,
)

# ── Otomatik yenileme ─────────────────────────────────────────────────────────
time.sleep(refresh)
st.rerun()
