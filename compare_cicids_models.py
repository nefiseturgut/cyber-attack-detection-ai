# -*- coding: utf-8 -*-
"""
CICIDS2018 - Tüm Modellerin Karşılaştırması
LSTM, LightGBM, XGBoost, CNN ve Ensemble modellerini karşılaştırır
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import tensorflow as tf
from tensorflow import keras
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def load_test_data():
    """Test verilerini yükle"""
    print("[*] Test verileri yukleniyor...")
    
    X_test = np.load('processed_data_cicids/X_test.npy')
    y_test = np.load('processed_data_cicids/y_test.npy')
    X_test_seq = np.load('lstm_data_cicids/X_test_seq.npy')
    y_test_seq = np.load('lstm_data_cicids/y_test_seq.npy')
    
    # Sequence uzunluğuna göre hizala
    seq_len = len(X_test_seq)
    X_test = X_test[:seq_len]
    y_test = y_test[:seq_len]
    
    print(f"  [OK] Tabular: {X_test.shape}, Sequence: {X_test_seq.shape}")
    return X_test, y_test, X_test_seq, y_test_seq

def evaluate_lstm(X_test_seq, y_test):
    """LSTM modelini değerlendir"""
    print("\n[*] LSTM modeli degerlendiriliyor...")
    try:
        model = keras.models.load_model('models/best_lstm_model_cicids.keras')
        y_proba = model.predict(X_test_seq, verbose=0).flatten()
        y_pred = (y_proba >= 0.5).astype(int)
        
        metrics = {
            'Model': 'LSTM',
            'Accuracy': accuracy_score(y_test, y_pred) * 100,
            'Precision': precision_score(y_test, y_pred) * 100,
            'Recall': recall_score(y_test, y_pred) * 100,
            'F1-Score': f1_score(y_test, y_pred) * 100,
            'AUC': roc_auc_score(y_test, y_proba)
        }
        print(f"  [OK] LSTM - Accuracy: {metrics['Accuracy']:.2f}%")
        return metrics
    except Exception as e:
        print(f"  [ERROR] LSTM yuklenemedi: {e}")
        return None

def evaluate_cnn(X_test_seq, y_test):
    """CNN modelini değerlendir"""
    print("\n[*] CNN modeli degerlendiriliyor...")
    try:
        model = keras.models.load_model('models/best_cnn_model_cicids.keras')
        X_cnn = X_test_seq.reshape(X_test_seq.shape[0], X_test_seq.shape[1], X_test_seq.shape[2], 1)
        y_proba = model.predict(X_cnn, verbose=0).flatten()
        y_pred = (y_proba >= 0.5).astype(int)
        
        metrics = {
            'Model': 'CNN',
            'Accuracy': accuracy_score(y_test, y_pred) * 100,
            'Precision': precision_score(y_test, y_pred) * 100,
            'Recall': recall_score(y_test, y_pred) * 100,
            'F1-Score': f1_score(y_test, y_pred) * 100,
            'AUC': roc_auc_score(y_test, y_proba)
        }
        print(f"  [OK] CNN - Accuracy: {metrics['Accuracy']:.2f}%")
        return metrics
    except Exception as e:
        print(f"  [ERROR] CNN yuklenemedi: {e}")
        return None

def evaluate_lightgbm(X_test, y_test):
    """LightGBM modelini değerlendir"""
    print("\n[*] LightGBM modeli degerlendiriliyor...")
    try:
        model = lgb.Booster(model_file='models/lightgbm_model_cicids.txt')
        y_proba = model.predict(X_test)
        y_pred = (y_proba >= 0.5).astype(int)
        
        metrics = {
            'Model': 'LightGBM',
            'Accuracy': accuracy_score(y_test, y_pred) * 100,
            'Precision': precision_score(y_test, y_pred) * 100,
            'Recall': recall_score(y_test, y_pred) * 100,
            'F1-Score': f1_score(y_test, y_pred) * 100,
            'AUC': roc_auc_score(y_test, y_proba)
        }
        print(f"  [OK] LightGBM - Accuracy: {metrics['Accuracy']:.2f}%")
        return metrics
    except Exception as e:
        print(f"  [ERROR] LightGBM yuklenemedi: {e}")
        return None

def evaluate_xgboost(X_test, y_test):
    """XGBoost modelini değerlendir"""
    print("\n[*] XGBoost modeli degerlendiriliyor...")
    try:
        model = xgb.Booster()
        model.load_model('models/xgboost_model_cicids.json')
        
        # Feature names yükle
        with open('processed_data_cicids/feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        dmatrix = xgb.DMatrix(X_test, feature_names=feature_names)
        y_proba = model.predict(dmatrix)
        y_pred = (y_proba >= 0.5).astype(int)
        
        metrics = {
            'Model': 'XGBoost',
            'Accuracy': accuracy_score(y_test, y_pred) * 100,
            'Precision': precision_score(y_test, y_pred) * 100,
            'Recall': recall_score(y_test, y_pred) * 100,
            'F1-Score': f1_score(y_test, y_pred) * 100,
            'AUC': roc_auc_score(y_test, y_proba)
        }
        print(f"  [OK] XGBoost - Accuracy: {metrics['Accuracy']:.2f}%")
        return metrics
    except Exception as e:
        print(f"  [ERROR] XGBoost yuklenemedi: {e}")
        return None

def evaluate_ensemble(X_test, X_test_seq, y_test):
    """Ensemble modelini değerlendir"""
    print("\n[*] Ensemble modeli degerlendiriliyor...")
    try:
        # Feature names
        with open('processed_data_cicids/feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        # Modelleri yükle
        lstm_model = keras.models.load_model('models/best_lstm_model_cicids.keras')
        cnn_model = keras.models.load_model('models/best_cnn_model_cicids.keras')
        lgbm_model = lgb.Booster(model_file='models/lightgbm_model_cicids.txt')
        xgb_model = xgb.Booster()
        xgb_model.load_model('models/xgboost_model_cicids.json')
        
        # Tahminler
        lstm_pred = lstm_model.predict(X_test_seq, verbose=0).flatten()
        X_cnn = X_test_seq.reshape(X_test_seq.shape[0], X_test_seq.shape[1], X_test_seq.shape[2], 1)
        cnn_pred = cnn_model.predict(X_cnn, verbose=0).flatten()
        lgbm_pred = lgbm_model.predict(X_test)
        dmatrix = xgb.DMatrix(X_test, feature_names=feature_names)
        xgb_pred = xgb_model.predict(dmatrix)
        
        # Ağırlıklı ortalama
        weights = {'lstm': 0.25, 'cnn': 0.15, 'lgbm': 0.30, 'xgb': 0.30}
        y_proba = (lstm_pred * weights['lstm'] + 
                   cnn_pred * weights['cnn'] + 
                   lgbm_pred * weights['lgbm'] + 
                   xgb_pred * weights['xgb'])
        y_pred = (y_proba >= 0.5).astype(int)
        
        metrics = {
            'Model': 'Ensemble',
            'Accuracy': accuracy_score(y_test, y_pred) * 100,
            'Precision': precision_score(y_test, y_pred) * 100,
            'Recall': recall_score(y_test, y_pred) * 100,
            'F1-Score': f1_score(y_test, y_pred) * 100,
            'AUC': roc_auc_score(y_test, y_proba)
        }
        print(f"  [OK] Ensemble - Accuracy: {metrics['Accuracy']:.2f}%")
        return metrics
    except Exception as e:
        print(f"  [ERROR] Ensemble degerlendirilemedi: {e}")
        return None

def create_comparison_chart(results_df):
    """Karşılaştırma grafiklerini oluştur"""
    print("\n[*] Karsilastirma grafikleri olusturuluyor...")
    
    # Renk paleti
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    # 2x2 subplot düzeni
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('CICIDS2018 - Tüm Modellerin Karşılaştırması', 
                 fontsize=20, fontweight='bold', y=0.995)
    
    # 1. Accuracy Karşılaştırması
    ax1 = axes[0, 0]
    bars1 = ax1.bar(results_df['Model'], results_df['Accuracy'], color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy Karşılaştırması', fontsize=14, fontweight='bold')
    ax1.set_ylim([90, 100])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    # Değerleri bar üzerine yaz
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 2. Precision, Recall, F1-Score Karşılaştırması
    ax2 = axes[0, 1]
    x = np.arange(len(results_df['Model']))
    width = 0.25
    ax2.bar(x - width, results_df['Precision'], width, label='Precision', color='#3498db', alpha=0.8, edgecolor='black')
    ax2.bar(x, results_df['Recall'], width, label='Recall', color='#2ecc71', alpha=0.8, edgecolor='black')
    ax2.bar(x + width, results_df['F1-Score'], width, label='F1-Score', color='#e74c3c', alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Skor (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Precision, Recall ve F1-Score', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(results_df['Model'])
    ax2.legend(loc='lower right', fontsize=10)
    ax2.set_ylim([80, 100])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 3. AUC Karşılaştırması
    ax3 = axes[1, 0]
    bars3 = ax3.bar(results_df['Model'], results_df['AUC'], color=colors, alpha=0.8, edgecolor='black')
    ax3.set_ylabel('AUC Skoru', fontsize=12, fontweight='bold')
    ax3.set_title('AUC Karşılaştırması', fontsize=14, fontweight='bold')
    ax3.set_ylim([0.90, 1.0])
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    # Değerleri bar üzerine yaz
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 4. Genel Performans Tablosu
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    
    # Tabloya renk ekle
    table_data = []
    for _, row in results_df.iterrows():
        table_data.append([
            row['Model'],
            f"{row['Accuracy']:.2f}%",
            f"{row['Precision']:.2f}%",
            f"{row['Recall']:.2f}%",
            f"{row['F1-Score']:.2f}%",
            f"{row['AUC']:.4f}"
        ])
    
    table = ax4.table(cellText=table_data,
                     colLabels=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.15, 0.17, 0.17, 0.17, 0.17, 0.17])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Header stillendirme
    for i in range(6):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Satır renkleri
    for i in range(1, len(results_df) + 1):
        for j in range(6):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
            else:
                table[(i, j)].set_facecolor('white')
    
    ax4.set_title('Detaylı Performans Metrikleri', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('models/cicids_comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    print("  [OK] models/cicids_comprehensive_comparison.png")
    plt.close()

def generate_report(results_df):
    """Metin raporu oluştur"""
    print("\n[*] Rapor olusturuluyor...")
    
    report = f"""
{'='*80}
CICIDS2018 - TÜM MODELLERİN KARŞILAŞTIRMASI
{'='*80}

PERFORMANS METRİKLERİ:
{'-'*80}
"""
    
    for _, row in results_df.iterrows():
        report += f"""
{row['Model']:15s}
  Accuracy:  {row['Accuracy']:6.2f}%
  Precision: {row['Precision']:6.2f}%
  Recall:    {row['Recall']:6.2f}%
  F1-Score:  {row['F1-Score']:6.2f}%
  AUC:       {row['AUC']:.4f}
{'-'*80}
"""
    
    # En iyi model
    best_model = results_df.loc[results_df['Accuracy'].idxmax()]
    report += f"""
EN İYİ MODEL: {best_model['Model']}
  Accuracy: {best_model['Accuracy']:.2f}%
  F1-Score: {best_model['F1-Score']:.2f}%
  AUC:      {best_model['AUC']:.4f}

{'='*80}
"""
    
    with open('models/cicids_comparison_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("  [OK] models/cicids_comparison_report.txt")
    print(report)

def main():
    print("="*80)
    print("*** CICIDS2018 - TÜM MODELLERİN KARŞILAŞTIRILMASI ***")
    print("="*80)
    
    # Veriyi yükle
    X_test, y_test, X_test_seq, y_test_seq = load_test_data()
    
    # Modelleri değerlendir
    results = []
    
    lstm_metrics = evaluate_lstm(X_test_seq, y_test_seq)
    if lstm_metrics:
        results.append(lstm_metrics)
    
    cnn_metrics = evaluate_cnn(X_test_seq, y_test_seq)
    if cnn_metrics:
        results.append(cnn_metrics)
    
    lightgbm_metrics = evaluate_lightgbm(X_test, y_test)
    if lightgbm_metrics:
        results.append(lightgbm_metrics)
    
    xgboost_metrics = evaluate_xgboost(X_test, y_test)
    if xgboost_metrics:
        results.append(xgboost_metrics)
    
    ensemble_metrics = evaluate_ensemble(X_test, X_test_seq, y_test_seq)
    if ensemble_metrics:
        results.append(ensemble_metrics)
    
    # DataFrame oluştur
    results_df = pd.DataFrame(results)
    
    # Karşılaştırma grafiği
    create_comparison_chart(results_df)
    
    # Rapor
    generate_report(results_df)
    
    print("\n[SUCCESS] Karsilastirma tamamlandi!")
    print(f"[INFO] Toplam {len(results)} model karsilastirildi.")

if __name__ == "__main__":
    main()
