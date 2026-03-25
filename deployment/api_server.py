"""
🚀 Siber Saldırı Tespit API Server
Flask tabanlı REST API - Model Deployment

Kullanım:
    python api_server.py

API Endpoints:
    GET  /                  - API ana sayfa
    GET  /health            - Health check
    GET  /models            - Mevcut modeller
    POST /predict           - Tekil tahmin
    POST /predict/batch     - Toplu tahmin
    GET  /stats             - Model istatistikleri
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from tensorflow import keras
import os
import time
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Cross-origin requests için

# ============================================================================
# MODEL YÜKLEME
# ============================================================================

class ModelServer:
    """Tüm modelleri yükler ve tahmin yapar"""
    
    def __init__(self):
        self.models = {}
        self.load_times = {}
        self.prediction_count = 0
        self.total_prediction_time = 0
        
        # Mevcut modelleri yükle
        self._load_models()
    
    def _load_models(self):
        """Tüm eğitilmiş modelleri yükle"""
        
        print("📦 Modeller yükleniyor...")
        
        # UNSW-NB15 modelleri (en iyi performans)
        model_configs = {
            'unsw_lstm': {
                'path': '../models/best_lstm_model_unsw.keras',
                'type': 'keras',
                'accuracy': 96.65,
                'description': 'LSTM model - Sequence-based'
            },
            'unsw_cnn': {
                'path': '../models/best_cnn_model_unsw.keras',
                'type': 'keras',
                'accuracy': 98.55,
                'description': 'CNN model - Best accuracy'
            },
            'unsw_lightgbm': {
                'path': '../models/lightgbm_model_unsw.txt',
                'type': 'lightgbm',
                'accuracy': 87.70,
                'description': 'LightGBM - Fastest prediction'
            },
            'unsw_xgboost': {
                'path': '../models/xgboost_model_unsw.json',
                'type': 'xgboost',
                'accuracy': 87.40,
                'description': 'XGBoost - Balanced performance'
            }
        }
        
        for model_name, config in model_configs.items():
            try:
                start_time = time.time()
                
                if config['type'] == 'keras' and os.path.exists(config['path']):
                    self.models[model_name] = keras.models.load_model(config['path'])
                    print(f"  ✅ {model_name} yüklendi (Keras)")
                    
                elif config['type'] == 'lightgbm' and os.path.exists(config['path']):
                    self.models[model_name] = lgb.Booster(model_file=config['path'])
                    print(f"  ✅ {model_name} yüklendi (LightGBM)")
                    
                elif config['type'] == 'xgboost' and os.path.exists(config['path']):
                    model = xgb.Booster()
                    model.load_model(config['path'])
                    self.models[model_name] = model
                    print(f"  ✅ {model_name} yüklendi (XGBoost)")
                
                self.load_times[model_name] = time.time() - start_time
                
            except Exception as e:
                print(f"  ❌ {model_name} yüklenemedi: {e}")
        
        print(f"\n✨ {len(self.models)} model başarıyla yüklendi!")
    
    def predict(self, data, model_name='unsw_cnn'):
        """Tahmin yap"""
        
        if model_name not in self.models:
            raise ValueError(f"Model bulunamadı: {model_name}")
        
        start_time = time.time()
        model = self.models[model_name]
        
        # Model türüne göre tahmin
        if isinstance(model, keras.Model):
            # Keras modeli - sequence formatı gerekli
            if len(data.shape) == 2:
                # (features,) -> (1, sequence_length, features)
                # UNSW için 42 feature var
                data = data.reshape(1, -1, data.shape[-1])
            prediction = model.predict(data, verbose=0)
            prediction_proba = float(prediction[0][0])
            
        elif isinstance(model, lgb.Booster):
            # LightGBM - tabular data
            if len(data.shape) == 3:
                data = data.reshape(data.shape[0], -1)
            prediction = model.predict(data)
            prediction_proba = float(prediction[0]) if len(prediction.shape) > 0 else float(prediction)
            
        elif isinstance(model, xgb.Booster):
            # XGBoost - DMatrix formatı
            if len(data.shape) == 3:
                data = data.reshape(data.shape[0], -1)
            dmatrix = xgb.DMatrix(data)
            prediction = model.predict(dmatrix)
            prediction_proba = float(prediction[0])
        
        prediction_time = time.time() - start_time
        
        # İstatistikleri güncelle
        self.prediction_count += 1
        self.total_prediction_time += prediction_time
        
        # Sonuç
        result = {
            'prediction': 'attack' if prediction_proba > 0.5 else 'normal',
            'confidence': float(prediction_proba) if prediction_proba > 0.5 else float(1 - prediction_proba),
            'probability_attack': float(prediction_proba),
            'probability_normal': float(1 - prediction_proba),
            'prediction_time_ms': round(prediction_time * 1000, 2),
            'model_used': model_name,
            'timestamp': datetime.now().isoformat()
        }
        
        return result

# Global model server instance
model_server = ModelServer()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/')
def home():
    """API ana sayfa"""
    return jsonify({
        'service': 'Siber Saldırı Tespit API',
        'version': '1.0',
        'status': 'running',
        'models_loaded': len(model_server.models),
        'endpoints': {
            'health': '/health',
            'models': '/models',
            'predict': '/predict [POST]',
            'batch_predict': '/predict/batch [POST]',
            'stats': '/stats'
        },
        'documentation': '/docs'
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(model_server.models),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/models')
def get_models():
    """Mevcut modellerin listesi"""
    models_info = []
    
    model_details = {
        'unsw_cnn': {'accuracy': 98.55, 'recall': 98.64, 'best_for': 'Yüksek doğruluk'},
        'unsw_lstm': {'accuracy': 96.65, 'recall': 97.00, 'best_for': 'Sequence patterns'},
        'unsw_lightgbm': {'accuracy': 87.70, 'recall': 98.80, 'best_for': 'Hız (2.6s)'},
        'unsw_xgboost': {'accuracy': 87.40, 'recall': 98.82, 'best_for': 'Dengeli performans'}
    }
    
    for model_name in model_server.models.keys():
        details = model_details.get(model_name, {})
        models_info.append({
            'name': model_name,
            'loaded': True,
            'load_time_ms': round(model_server.load_times[model_name] * 1000, 2),
            'accuracy': details.get('accuracy'),
            'recall': details.get('recall'),
            'best_for': details.get('best_for')
        })
    
    return jsonify({
        'total_models': len(models_info),
        'models': models_info,
        'recommended': 'unsw_cnn'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Tekil tahmin yap
    
    Request JSON:
    {
        "features": [0.1, 0.2, ...],  # 42 özellik (UNSW-NB15)
        "model": "unsw_cnn"  # opsiyonel, default: unsw_cnn
    }
    """
    try:
        data = request.get_json()
        
        if 'features' not in data:
            return jsonify({'error': 'features alanı gerekli'}), 400
        
        features = np.array(data['features'])
        model_name = data.get('model', 'unsw_cnn')
        
        # Feature sayısı kontrolü
        if features.shape[-1] != 42:
            return jsonify({
                'error': f'42 özellik bekleniyor, {features.shape[-1]} özellik alındı'
            }), 400
        
        # Tahmin yap
        result = model_server.predict(features, model_name)
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Toplu tahmin
    
    Request JSON:
    {
        "features": [[0.1, 0.2, ...], [0.3, 0.4, ...], ...],
        "model": "unsw_lightgbm"  # Toplu tahmin için hızlı model önerilir
    }
    """
    try:
        data = request.get_json()
        
        if 'features' not in data:
            return jsonify({'error': 'features alanı gerekli'}), 400
        
        features = np.array(data['features'])
        model_name = data.get('model', 'unsw_lightgbm')  # Batch için hızlı model
        
        results = []
        for i, feature_row in enumerate(features):
            try:
                result = model_server.predict(feature_row, model_name)
                result['index'] = i
                results.append(result)
            except Exception as e:
                results.append({
                    'index': i,
                    'error': str(e)
                })
        
        # Özet istatistikler
        successful = [r for r in results if 'error' not in r]
        attacks_detected = sum(1 for r in successful if r['prediction'] == 'attack')
        
        return jsonify({
            'success': True,
            'total_samples': len(features),
            'successful_predictions': len(successful),
            'attacks_detected': attacks_detected,
            'normal_detected': len(successful) - attacks_detected,
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/stats')
def stats():
    """API istatistikleri"""
    avg_prediction_time = (
        model_server.total_prediction_time / model_server.prediction_count
        if model_server.prediction_count > 0 else 0
    )
    
    return jsonify({
        'total_predictions': model_server.prediction_count,
        'average_prediction_time_ms': round(avg_prediction_time * 1000, 2),
        'models_loaded': len(model_server.models),
        'uptime': 'running',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/docs')
def docs():
    """API dokümantasyonu"""
    return """
    <html>
    <head><title>Siber Saldırı Tespit API Dokümantasyonu</title></head>
    <body style="font-family: Arial; padding: 20px; max-width: 900px; margin: auto;">
        <h1>🛡️ Siber Saldırı Tespit API</h1>
        <p>Bu API, eğitilmiş makine öğrenmesi modellerini kullanarak ağ trafiğindeki 
        siber saldırıları tespit eder.</p>
        
        <h2>Endpoints</h2>
        
        <h3>GET /models</h3>
        <p>Mevcut modellerin listesini döner.</p>
        <pre>curl http://localhost:5000/models</pre>
        
        <h3>POST /predict</h3>
        <p>Tekil tahmin yapar.</p>
        <pre>
curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"features": [0.1, 0.2, ..., 0.42], "model": "unsw_cnn"}'
        </pre>
        
        <h3>POST /predict/batch</h3>
        <p>Toplu tahmin yapar (hızlı modeller önerilir).</p>
        
        <h3>GET /stats</h3>
        <p>API kullanım istatistiklerini döner.</p>
        
        <h2>Önerilen Modeller</h2>
        <ul>
            <li><b>unsw_cnn</b>: En yüksek doğruluk (98.55%)</li>
            <li><b>unsw_lightgbm</b>: En hızlı (toplu tahmin için ideal)</li>
            <li><b>unsw_lstm</b>: Sequence pattern'leri için</li>
        </ul>
    </body>
    </html>
    """

# ============================================================================
# SERVER START
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 SİBER SALDIRI TESPİT API SERVER")
    print("="*80)
    print("\n📡 Server başlatılıyor...")
    print(f"🌐 URL: http://localhost:5000")
    print(f"📚 Dokümantasyon: http://localhost:5000/docs")
    print(f"💚 Health Check: http://localhost:5000/health")
    print("\n" + "="*80 + "\n")
    
    app.run(
        host='0.0.0.0',  # Tüm network interface'lerden erişilebilir
        port=5000,
        debug=True  # Development için, production'da False olmalı
    )
