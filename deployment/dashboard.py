"""
🎨 Siber Saldırı Tespit Dashboard
Streamlit tabanlı interaktif web arayüzü

Kullanım:
    streamlit run dashboard.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import requests
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# ============================================================================
# SAYFA AYARLARI
# ============================================================================

st.set_page_config(
    page_title="🛡️ Siber Saldırı Tespit Sistemi",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .attack-alert {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .normal-alert {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# YARDIMCI FONKSİYONLAR
# ============================================================================

def check_api_health(api_url):
    """API sağlık kontrolü"""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_available_models(api_url):
    """Mevcut modelleri getir"""
    try:
        response = requests.get(f"{api_url}/models", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def predict_single(api_url, features, model_name):
    """Tekil tahmin yap"""
    try:
        payload = {
            "features": features.tolist(),
            "model": model_name
        }
        response = requests.post(
            f"{api_url}/predict",
            json=payload,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Hata: {e}")
        return None

def get_api_stats(api_url):
    """API istatistikleri"""
    try:
        response = requests.get(f"{api_url}/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

# ============================================================================
# SESSION STATE
# ============================================================================

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

if 'api_url' not in st.session_state:
    st.session_state.api_url = "http://localhost:5000"

# ============================================================================
# HEADER
# ============================================================================

st.markdown('<h1 class="main-header">🛡️ Siber Saldırı Tespit Sistemi</h1>', 
            unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("⚙️ Ayarlar")
    
    # API URL
    api_url = st.text_input(
        "API URL",
        value=st.session_state.api_url,
        help="Flask API server adresi"
    )
    st.session_state.api_url = api_url
    
    # API Sağlık Kontrolü
    st.subheader("📡 API Durumu")
    if check_api_health(api_url):
        st.success("✅ API Aktif")
        
        # Mevcut modeller
        models_data = get_available_models(api_url)
        if models_data:
            st.info(f"🤖 {models_data['total_models']} model yüklü")
    else:
        st.error("❌ API'ye bağlanılamıyor")
        st.info("API'yi başlatmak için:\n```bash\npython deployment/api_server.py\n```")
    
    st.markdown("---")
    
    # Model Seçimi
    st.subheader("🎯 Model Seçimi")
    models_data = get_available_models(api_url)
    
    if models_data:
        model_options = {
            m['name']: f"{m['name']} (Acc: {m['accuracy']}%)"
            for m in models_data['models']
        }
        selected_model = st.selectbox(
            "Model",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x]
        )
    else:
        selected_model = "unsw_cnn"
        st.warning("Model listesi alınamadı")
    
    st.markdown("---")
    
    # İstatistikler
    st.subheader("📊 API İstatistikleri")
    stats = get_api_stats(api_url)
    if stats:
        st.metric("Toplam Tahmin", stats['total_predictions'])
        st.metric("Ort. Tahmin Süresi", 
                 f"{stats['average_prediction_time_ms']:.2f} ms")

# ============================================================================
# ANA İÇERİK
# ============================================================================

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Tekil Tahmin", 
    "📊 Toplu Tahmin", 
    "📈 Geçmiş",
    "ℹ️ Bilgi"
])

# ============================================================================
# TAB 1: TEKİL TAHMİN
# ============================================================================

with tab1:
    st.header("🎯 Tekil Ağ Trafiği Analizi")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Veri Girişi")
        
        # Veri giriş yöntemini seç
        input_method = st.radio(
            "Veri Giriş Yöntemi",
            ["Manuel Giriş", "Rastgele Veri", "Örnek Saldırı", "Örnek Normal"]
        )
        
        # 42 feature için örnek veri
        if input_method == "Rastgele Veri":
            features = np.random.randn(42)
            st.info("ℹ️ Rastgele veri oluşturuldu (test amaçlı)")
            
        elif input_method == "Örnek Saldırı":
            # Gerçek saldırı örneği benzeri
            features = np.random.randn(42)
            features[0] = 5.0  # Yüksek değerler
            features[5] = 10.0
            features[10] = 8.0
            st.warning("⚠️ Saldırı benzeri örnek veri")
            
        elif input_method == "Örnek Normal":
            # Normal trafik benzeri
            features = np.random.randn(42) * 0.1
            st.success("✅ Normal trafik benzeri örnek veri")
            
        else:  # Manuel Giriş
            st.info("💡 42 özellik değeri girin (virgülle ayrılmış)")
            manual_input = st.text_area(
                "Özellikler",
                value=",".join([f"{np.random.randn():.4f}" for _ in range(42)]),
                height=100
            )
            try:
                features = np.array([float(x.strip()) for x in manual_input.split(',')])
                if len(features) != 42:
                    st.error(f"❌ 42 özellik gerekli, {len(features)} girildi")
                    features = None
            except:
                st.error("❌ Geçersiz format")
                features = None
        
        # Tahmin butonu
        if st.button("🔍 Analiz Et", type="primary", use_container_width=True):
            if features is not None and len(features) == 42:
                with st.spinner("Analiz ediliyor..."):
                    result = predict_single(api_url, features, selected_model)
                    
                    if result and result.get('success'):
                        pred_result = result['result']
                        
                        # Sonucu session'a kaydet
                        st.session_state.prediction_history.append({
                            'timestamp': pred_result['timestamp'],
                            'prediction': pred_result['prediction'],
                            'confidence': pred_result['confidence'],
                            'model': selected_model
                        })
                        
                        # Sonuç göster
                        with col2:
                            st.subheader("📋 Sonuç")
                            
                            if pred_result['prediction'] == 'attack':
                                st.markdown(
                                    f"""
                                    <div class="attack-alert">
                                        <h2 style="color: #f44336;">⚠️ SALDIRI TESPİT EDİLDİ!</h2>
                                        <p>Güven: <b>{pred_result['confidence']*100:.2f}%</b></p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                            else:
                                st.markdown(
                                    f"""
                                    <div class="normal-alert">
                                        <h2 style="color: #4caf50;">✅ Normal Trafik</h2>
                                        <p>Güven: <b>{pred_result['confidence']*100:.2f}%</b></p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                            
                            # Detaylar
                            st.markdown("---")
                            st.metric("Tahmin Süresi", 
                                     f"{pred_result['prediction_time_ms']:.2f} ms")
                            st.metric("Kullanılan Model", selected_model)
                            
                            # Olasılık grafiği
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=['Normal', 'Saldırı'],
                                    y=[
                                        pred_result['probability_normal'],
                                        pred_result['probability_attack']
                                    ],
                                    marker_color=['green', 'red']
                                )
                            ])
                            fig.update_layout(
                                title="Olasılık Dağılımı",
                                yaxis_title="Olasılık",
                                height=300
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("❌ Tahmin yapılamadı")

# ============================================================================
# TAB 2: TOPLU TAHMİN
# ============================================================================

with tab2:
    st.header("📊 Toplu Ağ Trafiği Analizi")
    
    st.info("💡 CSV dosyası yükleyin veya örnek veri kullanın")
    
    # Dosya yükleme
    uploaded_file = st.file_uploader(
        "CSV Dosyası Yükle",
        type=['csv'],
        help="42 özellik içeren CSV dosyası"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_samples = st.slider("Örnek Veri Sayısı", 10, 1000, 100)
    
    with col2:
        if st.button("🎲 Örnek Veri Oluştur"):
            sample_data = np.random.randn(num_samples, 42)
            st.session_state.batch_data = sample_data
            st.success(f"✅ {num_samples} örnek oluşturuldu")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.batch_data = df.values
        st.success(f"✅ {len(df)} satır yüklendi")
    
    if 'batch_data' in st.session_state:
        st.write(f"Veri boyutu: {st.session_state.batch_data.shape}")
        
        if st.button("🚀 Toplu Analiz Başlat", type="primary"):
            with st.spinner("Toplu analiz yapılıyor..."):
                # API'ye gönder
                try:
                    payload = {
                        "features": st.session_state.batch_data.tolist(),
                        "model": selected_model
                    }
                    response = requests.post(
                        f"{api_url}/predict/batch",
                        json=payload,
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Sonuçları göster
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Toplam Örnek", result['total_samples'])
                        with col2:
                            st.metric("Saldırı Sayısı", 
                                     result['attacks_detected'],
                                     delta=None,
                                     delta_color="inverse")
                        with col3:
                            st.metric("Normal Sayısı", result['normal_detected'])
                        
                        # Grafik
                        fig = px.pie(
                            values=[result['attacks_detected'], result['normal_detected']],
                            names=['Saldırı', 'Normal'],
                            title="Tahmin Dağılımı",
                            color_discrete_sequence=['#f44336', '#4caf50']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Detaylı sonuçlar
                        with st.expander("📋 Detaylı Sonuçlar"):
                            results_df = pd.DataFrame(result['results'])
                            st.dataframe(results_df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Hata: {e}")

# ============================================================================
# TAB 3: GEÇMİŞ
# ============================================================================

with tab3:
    st.header("📈 Tahmin Geçmişi")
    
    if st.session_state.prediction_history:
        df_history = pd.DataFrame(st.session_state.prediction_history)
        
        # İstatistikler
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Toplam Tahmin", len(df_history))
        with col2:
            attacks = len(df_history[df_history['prediction'] == 'attack'])
            st.metric("Saldırı Sayısı", attacks)
        with col3:
            avg_conf = df_history['confidence'].mean()
            st.metric("Ortalama Güven", f"{avg_conf*100:.1f}%")
        
        # Zaman serisi grafiği
        if len(df_history) > 1:
            df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
            df_history['attack_numeric'] = (df_history['prediction'] == 'attack').astype(int)
            
            fig = px.scatter(
                df_history,
                x='timestamp',
                y='attack_numeric',
                color='prediction',
                size='confidence',
                title="Zaman Serisi Tahminler",
                color_discrete_map={'attack': 'red', 'normal': 'green'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Tablo
        st.dataframe(df_history, use_container_width=True)
        
        # Temizle butonu
        if st.button("🗑️ Geçmişi Temizle"):
            st.session_state.prediction_history = []
            st.rerun()
    else:
        st.info("📭 Henüz tahmin yapılmadı")

# ============================================================================
# TAB 4: BİLGİ
# ============================================================================

with tab4:
    st.header("ℹ️ Sistem Bilgileri")
    
    st.markdown("""
    ### 🛡️ Siber Saldırı Tespit Sistemi
    
    Bu dashboard, eğitilmiş makine öğrenmesi modellerini kullanarak
    ağ trafiğindeki siber saldırıları tespit eder.
    
    #### 🤖 Mevcut Modeller:
    
    - **CNN (Convolutional Neural Network)**
      - En yüksek accuracy: %98.55
      - Önerilen kullanım: Yüksek doğruluk gerekli
      
    - **LSTM (Long Short-Term Memory)**
      - Accuracy: %96.65
      - Önerilen kullanım: Sequence pattern'leri
      
    - **LightGBM**
      - Accuracy: %87.70
      - En hızlı: 2.6 saniye eğitim
      - Önerilen kullanım: Gerçek zamanlı sistemler
      
    - **XGBoost**
      - Accuracy: %87.40
      - Dengeli performans
    
    #### 📊 Dataset: UNSW-NB15
    - 42 özellik
    - Binary classification (Normal vs Saldırı)
    - Dengeli ve gerçekçi ağ trafiği
    
    #### 🚀 Kullanım:
    1. API server'ı başlatın
    2. Model seçin
    3. Veri girin veya yükleyin
    4. Analiz sonuçlarını görüntüleyin
    
    ---
    
    **Geliştirici:** Nefise  
    **Versiyon:** 1.0  
    **Tarih:** Ocak 2026
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666;">
        🛡️ Siber Saldırı Tespit Sistemi v1.0 | 
        Developed with ❤️ using Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
