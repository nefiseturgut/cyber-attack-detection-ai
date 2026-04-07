"""
Model Karşılaştırma: LSTM vs LightGBM
İki modelin performansını karşılaştırma ve analiz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time
import os


class ModelComparison:
    """LSTM ve LightGBM modellerini karşılaştır"""
    
    def __init__(self):
        self.lstm_model = None
        self.lgbm_model = None
        self.results = {}
        
    def load_models(self):
        """Model ve veriyi yükle"""
        print("\n" + "="*80)
        print("📦 MODELLERİ YÜKLEME")
        print("="*80)
        
        # LSTM modeli yükle
        print("\n🧠 LSTM modeli yükleniyor...")
        self.lstm_model = keras.models.load_model('models/best_lstm_model.keras')
        print("✅ LSTM yüklendi")
        
        # LightGBM modeli yükle
        print("\n⚡ LightGBM modeli yükleniyor...")
        self.lgbm_model = lgb.Booster(model_file='models/lightgbm_model.txt')
        print("✅ LightGBM yüklendi")
        
    def load_test_data(self):
        """Test verilerini yükle"""
        print("\n📊 Test verileri yükleniyor...")
        
        # LightGBM için normal veri
        X_test_lgbm = np.load('processed_data/X_test.npy')
        y_test = np.load('processed_data/y_test.npy')
        
        # LSTM için sequence veri
        X_test_lstm = np.load('lstm_data/X_test_seq.npy')
        y_test_lstm = np.load('lstm_data/y_test_seq.npy')
        
        print(f"✅ LightGBM test: {X_test_lgbm.shape}")
        print(f"✅ LSTM test: {X_test_lstm.shape}")
        
        return X_test_lgbm, X_test_lstm, y_test, y_test_lstm
    
    def evaluate_lstm(self, X_test, y_test):
        """LSTM modelini değerlendir"""
        print("\n" + "="*80)
        print("🧠 LSTM PERFORMANS DEĞERLENDİRMESİ")
        print("="*80)
        
        # Tahmin zamanı
        start_time = time.time()
        y_pred_proba = self.lstm_model.predict(X_test, verbose=0).flatten()
        prediction_time = time.time() - start_time
        
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Metrikler
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba),
            'prediction_time': prediction_time,
            'samples': len(y_test)
        }
        
        print(f"\n📈 LSTM Sonuçları:")
        print(f"   Accuracy:  {metrics['accuracy']*100:.2f}%")
        print(f"   Precision: {metrics['precision']*100:.2f}%")
        print(f"   Recall:    {metrics['recall']*100:.2f}%")
        print(f"   F1-Score:  {metrics['f1_score']*100:.2f}%")
        print(f"   AUC:       {metrics['auc']:.4f}")
        print(f"   Tahmin süresi: {prediction_time:.2f} saniye")
        print(f"   Örnek/saniye: {len(y_test)/prediction_time:.0f}")
        
        self.results['LSTM'] = metrics
        return metrics
    
    def evaluate_lgbm(self, X_test, y_test):
        """LightGBM modelini değerlendir"""
        print("\n" + "="*80)
        print("⚡ LIGHTGBM PERFORMANS DEĞERLENDİRMESİ")
        print("="*80)
        
        # Tahmin zamanı
        start_time = time.time()
        y_pred_proba = self.lgbm_model.predict(X_test)
        prediction_time = time.time() - start_time
        
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Metrikler
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba),
            'prediction_time': prediction_time,
            'samples': len(y_test)
        }
        
        print(f"\n📈 LightGBM Sonuçları:")
        print(f"   Accuracy:  {metrics['accuracy']*100:.2f}%")
        print(f"   Precision: {metrics['precision']*100:.2f}%")
        print(f"   Recall:    {metrics['recall']*100:.2f}%")
        print(f"   F1-Score:  {metrics['f1_score']*100:.2f}%")
        print(f"   AUC:       {metrics['auc']:.4f}")
        print(f"   Tahmin süresi: {prediction_time:.2f} saniye")
        print(f"   Örnek/saniye: {len(y_test)/prediction_time:.0f}")
        
        self.results['LightGBM'] = metrics
        return metrics
    
    def compare_models(self):
        """Modelleri karşılaştır"""
        print("\n" + "="*80)
        print("🔄 MODEL KARŞILAŞTIRMASI")
        print("="*80)
        
        # Karşılaştırma tablosu
        comparison_df = pd.DataFrame(self.results).T
        
        print(f"\n📊 Performans Karşılaştırması:")
        print("\n" + "="*80)
        print(f"{'Metrik':<15} {'LSTM':>15} {'LightGBM':>15} {'Kazanan':>15}")
        print("="*80)
        
        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        
        for metric in metrics_to_compare:
            lstm_val = self.results['LSTM'][metric]
            lgbm_val = self.results['LightGBM'][metric]
            
            if metric == 'prediction_time':
                winner = '⚡ LightGBM' if lgbm_val < lstm_val else '🧠 LSTM'
            else:
                winner = '🧠 LSTM' if lstm_val > lgbm_val else '⚡ LightGBM'
                if abs(lstm_val - lgbm_val) < 0.001:
                    winner = '🤝 Eşit'
            
            print(f"{metric.capitalize():<15} {lstm_val:>15.4f} {lgbm_val:>15.4f} {winner:>15}")
        
        # Hız karşılaştırması
        print("\n" + "="*80)
        print(f"{'Hız Metrikleri':<15} {'LSTM':>15} {'LightGBM':>15} {'Fark':>15}")
        print("="*80)
        
        lstm_time = self.results['LSTM']['prediction_time']
        lgbm_time = self.results['LightGBM']['prediction_time']
        speedup = lstm_time / lgbm_time
        
        print(f"{'Tahmin süresi':<15} {lstm_time:>14.2f}s {lgbm_time:>14.2f}s {speedup:>13.1f}x")
        
        lstm_throughput = self.results['LSTM']['samples'] / lstm_time
        lgbm_throughput = self.results['LightGBM']['samples'] / lgbm_time
        
        print(f"{'Örnek/saniye':<15} {lstm_throughput:>14.0f} {lgbm_throughput:>14.0f} {lgbm_throughput/lstm_throughput:>13.1f}x")
        
        return comparison_df
    
    def plot_comparison(self, save_path='models/model_comparison.png'):
        """Karşılaştırma grafiği"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('LSTM vs LightGBM - Model Karşılaştırması', fontsize=16, fontweight='bold')
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        colors = ['#2ecc71', '#3498db']
        
        # Metrik karşılaştırma
        ax = axes[0, 0]
        x = np.arange(len(metrics))
        width = 0.35
        
        lstm_vals = [self.results['LSTM'][m] for m in metrics]
        lgbm_vals = [self.results['LightGBM'][m] for m in metrics]
        
        ax.bar(x - width/2, lstm_vals, width, label='LSTM', color=colors[0], alpha=0.8)
        ax.bar(x + width/2, lgbm_vals, width, label='LightGBM', color=colors[1], alpha=0.8)
        
        ax.set_ylabel('Skor', fontweight='bold')
        ax.set_title('Performans Metrikleri', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in metrics], rotation=15)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0.9, 1.0])
        
        # AUC karşılaştırma
        ax = axes[0, 1]
        auc_vals = [self.results['LSTM']['auc'], self.results['LightGBM']['auc']]
        bars = ax.bar(['LSTM', 'LightGBM'], auc_vals, color=colors, alpha=0.8)
        ax.set_ylabel('AUC Skoru', fontweight='bold')
        ax.set_title('AUC Karşılaştırması', fontweight='bold')
        ax.set_ylim([0.9, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Hız karşılaştırması
        ax = axes[1, 0]
        time_vals = [self.results['LSTM']['prediction_time'], 
                    self.results['LightGBM']['prediction_time']]
        bars = ax.bar(['LSTM', 'LightGBM'], time_vals, color=colors, alpha=0.8)
        ax.set_ylabel('Saniye', fontweight='bold')
        ax.set_title('Tahmin Süresi (Düşük = İyi)', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}s', ha='center', va='bottom', fontweight='bold')
        
        # Throughput karşılaştırması
        ax = axes[1, 1]
        throughput_vals = [
            self.results['LSTM']['samples'] / self.results['LSTM']['prediction_time'],
            self.results['LightGBM']['samples'] / self.results['LightGBM']['prediction_time']
        ]
        bars = ax.bar(['LSTM', 'LightGBM'], throughput_vals, color=colors, alpha=0.8)
        ax.set_ylabel('Örnek/Saniye', fontweight='bold')
        ax.set_title('İşlem Hızı (Yüksek = İyi)', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n📊 Karşılaştırma grafiği kaydedildi: {save_path}")
    
    def generate_summary_report(self, save_path='models/comparison_report.txt'):
        """Özet rapor oluştur"""
        report = []
        report.append("="*80)
        report.append("MODEL KARŞILAŞTIRMA RAPORU: LSTM vs LightGBM")
        report.append("="*80)
        report.append("")
        
        # Performans özeti
        report.append("📊 PERFORMANS ÖZETİ")
        report.append("-"*80)
        
        lstm = self.results['LSTM']
        lgbm = self.results['LightGBM']
        
        report.append(f"\nLSTM Modeli:")
        report.append(f"  Accuracy:  {lstm['accuracy']*100:.2f}%")
        report.append(f"  Precision: {lstm['precision']*100:.2f}%")
        report.append(f"  Recall:    {lstm['recall']*100:.2f}%")
        report.append(f"  F1-Score:  {lstm['f1_score']*100:.2f}%")
        report.append(f"  AUC:       {lstm['auc']:.4f}")
        
        report.append(f"\nLightGBM Modeli:")
        report.append(f"  Accuracy:  {lgbm['accuracy']*100:.2f}%")
        report.append(f"  Precision: {lgbm['precision']*100:.2f}%")
        report.append(f"  Recall:    {lgbm['recall']*100:.2f}%")
        report.append(f"  F1-Score:  {lgbm['f1_score']*100:.2f}%")
        report.append(f"  AUC:       {lgbm['auc']:.4f}")
        
        # Hız karşılaştırması
        report.append(f"\n⚡ HIZ KARŞILAŞTIRMASI")
        report.append("-"*80)
        report.append(f"LSTM Tahmin Süresi:     {lstm['prediction_time']:.2f} saniye")
        report.append(f"LightGBM Tahmin Süresi: {lgbm['prediction_time']:.2f} saniye")
        report.append(f"Hız Farkı: LightGBM {lstm['prediction_time']/lgbm['prediction_time']:.1f}x daha hızlı")
        
        # Sonuç ve öneri
        report.append(f"\n💡 SONUÇ VE ÖNERİLER")
        report.append("-"*80)
        
        if lstm['accuracy'] > lgbm['accuracy']:
            report.append("✅ LSTM, accuracy açısından daha iyi performans gösteriyor.")
        else:
            report.append("✅ LightGBM, accuracy açısından daha iyi performans gösteriyor.")
        
        if lgbm['prediction_time'] < lstm['prediction_time']:
            report.append("⚡ LightGBM, tahmin hızında önemli ölçüde daha hızlı.")
        
        report.append("\n📌 Kullanım Önerileri:")
        report.append("  - Yüksek doğruluk gerekiyorsa: LSTM kullanın")
        report.append("  - Hızlı tahmin gerekiyorsa: LightGBM kullanın")
        report.append("  - Gerçek zamanlı sistemler için: LightGBM önerilir")
        report.append("  - Sequence pattern'leri önemliyse: LSTM önerilir")
        
        report.append("")
        report.append("="*80)
        
        # Dosyaya yaz
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"\n📄 Rapor kaydedildi: {save_path}")
        
        # Ekrana da yazdır
        print('\n'.join(report))


def main():
    """Ana fonksiyon - Model karşılaştırması"""
    print("\n" + "="*80)
    print("🔬 MODEL KARŞILAŞTIRMA ANALİZİ")
    print("="*80 + "\n")
    
    # Karşılaştırma objesi
    comparison = ModelComparison()
    
    # Modelleri yükle
    comparison.load_models()
    
    # Test verilerini yükle
    X_test_lgbm, X_test_lstm, y_test_lgbm, y_test_lstm = comparison.load_test_data()
    
    # LSTM değerlendir
    comparison.evaluate_lstm(X_test_lstm, y_test_lstm)
    
    # LightGBM değerlendir
    comparison.evaluate_lgbm(X_test_lgbm, y_test_lgbm)
    
    # Karşılaştır
    comparison_df = comparison.compare_models()
    
    # Görselleştir
    comparison.plot_comparison()
    
    # Rapor oluştur
    comparison.generate_summary_report()
    
    print("\n" + "="*80)
    print("✨ MODEL KARŞILAŞTIRMASI TAMAMLANDI!")
    print("="*80)
    
    return comparison


if __name__ == "__main__":
    comparison = main()
