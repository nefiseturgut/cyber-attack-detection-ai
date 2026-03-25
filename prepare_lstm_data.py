"""
LSTM için Veri Hazırlama
Ağ trafiği verilerini zaman dizisi formatına dönüştürme
"""

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import os


class LSTMDataPreparator:
    """LSTM modeli için veri hazırlama sınıfı"""
    
    def __init__(self, sequence_length=10):
        """
        Args:
            sequence_length: Her sequence'te kaç zaman adımı olacak
        """
        self.sequence_length = sequence_length
        
    def create_sequences(self, X, y):
        """
        Veriyi LSTM için sequence formatına dönüştür
        
        Args:
            X: Özellikler (samples, features)
            y: Etiketler (samples,)
            
        Returns:
            X_seq: (samples, sequence_length, features)
            y_seq: (samples,)
        """
        print(f"\n📦 Sequence oluşturuluyor (sequence_length={self.sequence_length})...")
        
        X_seq = []
        y_seq = []
        
        # Her sequence için
        for i in range(len(X) - self.sequence_length + 1):
            # Son sequence_length kadar örneği al
            seq_x = X[i:i + self.sequence_length]
            # Son etiket sequence'in etiketi olacak
            seq_y = y[i + self.sequence_length - 1]
            
            X_seq.append(seq_x)
            y_seq.append(seq_y)
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        print(f"✅ Sequence oluşturuldu:")
        print(f"   Girdi shape: {X.shape} -> {X_seq.shape}")
        print(f"   Etiket shape: {y.shape} -> {y_seq.shape}")
        
        return X_seq, y_seq
    
    def prepare_lstm_data(self, input_dir='processed_data', output_dir='lstm_data'):
        """
        İşlenmiş veriyi yükle ve LSTM formatına dönüştür
        """
        print("\n" + "="*80)
        print("🔄 LSTM VERİ HAZIRLIĞI")
        print("="*80)
        
        # Veriyi yükle
        print("\n📂 Veriler yükleniyor...")
        X_train = np.load(os.path.join(input_dir, 'X_train.npy'))
        y_train = np.load(os.path.join(input_dir, 'y_train.npy'))
        X_test = np.load(os.path.join(input_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(input_dir, 'y_test.npy'))
        
        print(f"✅ Orijinal veri yüklendi:")
        print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Sequence oluştur
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train)
        X_test_seq, y_test_seq = self.create_sequences(X_test, y_test)
        
        # Sınıf dağılımını göster
        print(f"\n📊 Sınıf Dağılımı:")
        train_normal = (y_train_seq == 0).sum()
        train_attack = (y_train_seq == 1).sum()
        test_normal = (y_test_seq == 0).sum()
        test_attack = (y_test_seq == 1).sum()
        
        print(f"   Train - Normal: {train_normal:,} ({train_normal/len(y_train_seq)*100:.1f}%), "
              f"Saldırı: {train_attack:,} ({train_attack/len(y_train_seq)*100:.1f}%)")
        print(f"   Test - Normal: {test_normal:,} ({test_normal/len(y_test_seq)*100:.1f}%), "
              f"Saldırı: {test_attack:,} ({test_attack/len(y_test_seq)*100:.1f}%)")
        
        # Kaydet
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, 'X_train_seq.npy'), X_train_seq)
        np.save(os.path.join(output_dir, 'y_train_seq.npy'), y_train_seq)
        np.save(os.path.join(output_dir, 'X_test_seq.npy'), X_test_seq)
        np.save(os.path.join(output_dir, 'y_test_seq.npy'), y_test_seq)
        
        # Metadata kaydet
        metadata = {
            'sequence_length': self.sequence_length,
            'n_features': X_train_seq.shape[2],
            'train_samples': X_train_seq.shape[0],
            'test_samples': X_test_seq.shape[0]
        }
        
        np.save(os.path.join(output_dir, 'metadata.npy'), metadata)
        
        print(f"\n💾 Veriler kaydedildi: '{output_dir}/'")
        print(f"   - X_train_seq.npy: {X_train_seq.shape}")
        print(f"   - y_train_seq.npy: {y_train_seq.shape}")
        print(f"   - X_test_seq.npy: {X_test_seq.shape}")
        print(f"   - y_test_seq.npy: {y_test_seq.shape}")
        print(f"   - metadata.npy")
        
        return X_train_seq, y_train_seq, X_test_seq, y_test_seq, metadata


def main():
    """Ana fonksiyon"""
    print("\n" + "="*80)
    print("🚀 LSTM VERİ HAZIRLAMA")
    print("="*80 + "\n")
    
    # Sequence length: Kaç zaman adımını birlikte değerlendireceğiz
    # Ağ trafiğinde bir pattern tespit etmek için 10-20 paket yeterli
    sequence_length = 10
    
    preparator = LSTMDataPreparator(sequence_length=sequence_length)
    X_train_seq, y_train_seq, X_test_seq, y_test_seq, metadata = preparator.prepare_lstm_data()
    
    print("\n" + "="*80)
    print("✨ LSTM VERİSİ HAZIR!")
    print("="*80)
    print(f"\n📊 Her sequence {sequence_length} zaman adımından oluşuyor")
    print(f"   Bu, ağdaki {sequence_length} ardışık paketi birlikte analiz etmek anlamına gelir.")
    print(f"\n✅ Şimdi LSTM modeli eğitilebilir!")
    
    return X_train_seq, y_train_seq, X_test_seq, y_test_seq, metadata


if __name__ == "__main__":
    X_train_seq, y_train_seq, X_test_seq, y_test_seq, metadata = main()
