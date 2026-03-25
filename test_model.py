"""
LSTM Modelini Test Etme ve Gerçek Zamanlı Saldırı Tespiti
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd


class CyberAttackDetector:
    """Eğitilmiş LSTM modeli ile saldırı tespiti"""
    
    def __init__(self, model_path='models/best_lstm_model.keras'):
        """
        Args:
            model_path: Eğitilmiş model dosyasının yolu
        """
        print("🔐 Siber Saldırı Tespit Sistemi Başlatılıyor...")
        
        # Modeli yükle
        self.model = keras.models.load_model(model_path)
        print(f"✅ Model yüklendi: {model_path}")
        
        # Metadata yükle
        metadata = np.load('lstm_data/metadata.npy', allow_pickle=True).item()
        self.sequence_length = metadata['sequence_length']
        self.n_features = metadata['n_features']
        
        print(f"📊 Model özellikleri:")
        print(f"   Sequence length: {self.sequence_length}")
        print(f"   Feature count: {self.n_features}")
        
        # Feature isimleri
        with open('processed_data/feature_names.txt', 'r') as f:
            self.feature_names = [line.strip() for line in f.readlines()]
    
    def predict_single(self, sequence):
        """
        Tek bir sequence için tahmin yap
        
        Args:
            sequence: (sequence_length, n_features) boyutunda numpy array
            
        Returns:
            prediction: 0 (Normal) veya 1 (Saldırı)
            confidence: Tahmin güven skoru (0-1)
        """
        # Reshape for model input
        sequence = sequence.reshape(1, self.sequence_length, self.n_features)
        
        # Tahmin yap
        prediction_proba = self.model.predict(sequence, verbose=0)[0][0]
        prediction = 1 if prediction_proba > 0.5 else 0
        
        return prediction, prediction_proba
    
    def predict_batch(self, sequences):
        """
        Birden fazla sequence için tahmin yap
        
        Args:
            sequences: (n_samples, sequence_length, n_features) boyutunda array
            
        Returns:
            predictions: Tahminler (n_samples,)
            confidences: Güven skorları (n_samples,)
        """
        prediction_probas = self.model.predict(sequences, verbose=0).flatten()
        predictions = (prediction_probas > 0.5).astype(int)
        
        return predictions, prediction_probas
    
    def analyze_traffic(self, traffic_sequence, show_details=True):
        """
        Ağ trafiği sequence'ini analiz et
        
        Args:
            traffic_sequence: Analiz edilecek traffic sequence
            show_details: Detayları göster
        """
        prediction, confidence = self.predict_single(traffic_sequence)
        
        if show_details:
            print("\n" + "="*80)
            print("🔍 TRAFIK ANALİZİ")
            print("="*80)
            
            if prediction == 1:
                print(f"\n⚠️  SALDIRI TESPİT EDİLDİ!")
                print(f"   Güven skoru: {confidence*100:.2f}%")
                print(f"   Risk seviyesi: {'YÜKSEK' if confidence > 0.8 else 'ORTA'}")
            else:
                print(f"\n✅ Normal Trafik")
                print(f"   Güven skoru: {(1-confidence)*100:.2f}%")
            
            print(f"\n📊 Sequence bilgileri:")
            print(f"   Zaman adımları: {self.sequence_length}")
            print(f"   Özellik sayısı: {self.n_features}")
        
        return prediction, confidence
    
    def test_on_test_set(self, n_samples=100):
        """
        Test setinden örnekler alıp model performansını göster
        
        Args:
            n_samples: Test edilecek örnek sayısı
        """
        print("\n" + "="*80)
        print("🧪 TEST SETİ ANALİZİ")
        print("="*80)
        
        # Test verisini yükle
        X_test = np.load('lstm_data/X_test_seq.npy')
        y_test = np.load('lstm_data/y_test_seq.npy')
        
        # Rastgele örnekler seç
        indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
        X_sample = X_test[indices]
        y_sample = y_test[indices]
        
        # Tahmin yap
        predictions, confidences = self.predict_batch(X_sample)
        
        # Sonuçları analiz et
        correct = (predictions == y_sample).sum()
        accuracy = correct / len(y_sample)
        
        # Detaylı analiz
        true_positives = ((predictions == 1) & (y_sample == 1)).sum()
        true_negatives = ((predictions == 0) & (y_sample == 0)).sum()
        false_positives = ((predictions == 1) & (y_sample == 0)).sum()
        false_negatives = ((predictions == 0) & (y_sample == 1)).sum()
        
        print(f"\n📊 {n_samples} örnek üzerinde test:")
        print(f"\n✅ Doğruluk: {accuracy*100:.2f}% ({correct}/{len(y_sample)})")
        
        print(f"\n📈 Detaylı Sonuçlar:")
        print(f"   True Positives (Saldırı→Saldırı):  {true_positives:4}")
        print(f"   True Negatives (Normal→Normal):    {true_negatives:4}")
        print(f"   False Positives (Normal→Saldırı):  {false_positives:4}")
        print(f"   False Negatives (Saldırı→Normal):  {false_negatives:4}")
        
        if true_positives + false_positives > 0:
            precision = true_positives / (true_positives + false_positives)
            print(f"\n   Precision: {precision*100:.2f}%")
        
        if true_positives + false_negatives > 0:
            recall = true_positives / (true_positives + false_negatives)
            print(f"   Recall: {recall*100:.2f}%")
        
        # Bazı örnekleri göster
        print(f"\n📋 Örnek Tahminler:")
        for i in range(min(5, len(predictions))):
            actual = "Saldırı" if y_sample[i] == 1 else "Normal"
            pred = "Saldırı" if predictions[i] == 1 else "Normal"
            conf = confidences[i] * 100
            status = "✅" if predictions[i] == y_sample[i] else "❌"
            
            print(f"   {status} Gerçek: {actual:8} | Tahmin: {pred:8} | Güven: {conf:5.1f}%")
        
        return accuracy, predictions, y_sample


def main():
    """Ana fonksiyon - Model testi"""
    print("\n" + "="*80)
    print("🛡️  LSTM SİBER SALDIRI TESPİT SİSTEMİ")
    print("="*80 + "\n")
    
    # Detector oluştur
    detector = CyberAttackDetector()
    
    # Test setinde performans testi
    print("\n" + "="*80)
    print("📊 Model Performans Testi")
    print("="*80)
    
    accuracy, preds, actuals = detector.test_on_test_set(n_samples=1000)
    
    # Gerçek test seti üzerinde tam değerlendirme
    print("\n" + "="*80)
    print("🎯 TAM TEST SETİ DEĞERLENDİRMESİ")
    print("="*80)
    
    X_test_full = np.load('lstm_data/X_test_seq.npy')
    y_test_full = np.load('lstm_data/y_test_seq.npy')
    
    print(f"\nToplam test örnekleri: {len(X_test_full):,}")
    
    # Batch batch tahmin yap (bellek için)
    batch_size = 1000
    all_predictions = []
    
    for i in range(0, len(X_test_full), batch_size):
        batch_x = X_test_full[i:i+batch_size]
        preds_batch, _ = detector.predict_batch(batch_x)
        all_predictions.extend(preds_batch)
    
    all_predictions = np.array(all_predictions)
    
    # Genel accuracy
    accuracy_full = (all_predictions == y_test_full).mean()
    
    print(f"\n✨ GENEL PERFORMANS:")
    print(f"   Doğruluk: {accuracy_full*100:.2f}%")
    
    # Confusion matrix
    tp = ((all_predictions == 1) & (y_test_full == 1)).sum()
    tn = ((all_predictions == 0) & (y_test_full == 0)).sum()
    fp = ((all_predictions == 1) & (y_test_full == 0)).sum()
    fn = ((all_predictions == 0) & (y_test_full == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n   Precision: {precision*100:.2f}%")
    print(f"   Recall:    {recall*100:.2f}%")
    print(f"   F1-Score:  {f1*100:.2f}%")
    
    print(f"\n   True Positives:  {tp:,}")
    print(f"   True Negatives:  {tn:,}")
    print(f"   False Positives: {fp:,}")
    print(f"   False Negatives: {fn:,}")
    
    print("\n" + "="*80)
    print("✅ MODEL HAZIR - Ağ trafiğini izleyebilir!")
    print("="*80)
    
    return detector


if __name__ == "__main__":
    detector = main()
