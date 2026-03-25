"""
Siber Saldırı Tespit Projesi - CICIDS2018 Veri Ön İşleme
CICIDS2018 veri seti için veri hazırlama scripti
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path


class CICIDS2018Preprocessor:
    """CICIDS2018 verisi için ön işleme sınıfı"""
    
    def __init__(self, data_dir='datasets/CICIDS2018'):
        self.data_dir = Path(data_dir)
        self.data = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def load_data(self, csv_file='Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv'):
        """Veri setini yükle"""
        print("📂 CICIDS2018 verisi yükleniyor...")
        
        csv_path = self.data_dir / csv_file
        self.data = pd.read_csv(csv_path, low_memory=False)
        print(f"✅ Veri yüklendi: {self.data.shape}")
        
        return self
    
    def explore_data(self):
        """Veriyi keşfet ve temel istatistikleri göster"""
        print("\n" + "="*80)
        print("📊 VERİ KEŞFİ - CICIDS2018")
        print("="*80)
        
        print(f"\nSatır sayısı: {self.data.shape[0]:,}")
        print(f"Sütun sayısı: {self.data.shape[1]}")
        
        print("\n--- İlk 5 Sütun ---")
        print(self.data.columns[:10].tolist())
        
        print("\n--- Veri Tipleri ---")
        print(self.data.dtypes.value_counts())
        
        print("\n--- Eksik Değerler ---")
        missing = self.data.isnull().sum()
        if missing.sum() > 0:
            print(f"Toplam eksik değer: {missing.sum():,}")
            print(missing[missing > 0].head(10))
        else:
            print("✅ Eksik değer yok!")
        
        if 'Label' in self.data.columns:
            print("\n--- Saldırı Türleri Dağılımı ---")
            label_counts = self.data['Label'].value_counts()
            print(label_counts)
            
            # Normal vs Saldırı
            benign_count = (self.data['Label'] == 'Benign').sum()
            attack_count = (self.data['Label'] != 'Benign').sum()
            print(f"\nBenign trafik: {benign_count:,} ({benign_count/len(self.data)*100:.1f}%)")
            print(f"Saldırı trafiği: {attack_count:,} ({attack_count/len(self.data)*100:.1f}%)")
        
        return self
    
    def clean_data(self):
        """Veriyi temizle"""
        print("\n" + "="*80)
        print("🧹 VERİ TEMİZLEME")
        print("="*80)
        
        initial_shape = self.data.shape
        
        # Infinity ve NaN değerlerini temizle
        print("\n🔍 Infinity ve NaN değerleri kontrol ediliyor...")
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # NaN satırları kaldır
        nan_count = self.data.isnull().sum().sum()
        if nan_count > 0:
            self.data.dropna(inplace=True)
            print(f"✅ {nan_count:,} NaN değeri içeren satırlar kaldırıldı")
        
        # Duplikatları kontrol et
        duplicates = self.data.duplicated().sum()
        print(f"\nDuplikat sayısı: {duplicates:,}")
        
        if duplicates > 0:
            self.data = self.data.drop_duplicates()
            print(f"✅ Duplikatlar kaldırıldı")
        
        print(f"\nVeri boyutu: {initial_shape} → {self.data.shape}")
        
        return self
    
    def create_binary_labels(self):
        """İkili sınıflandırma için etiketler oluştur (Benign vs Attack)"""
        print("\n" + "="*80)
        print("🏷️  ETİKET DÖNÜŞÜMÜ")
        print("="*80)
        
        if 'Label' not in self.data.columns:
            raise ValueError("'Label' sütunu bulunamadı!")
        
        # Binary sınıflandırma için yeni sütun oluştur
        self.data['is_attack'] = (self.data['Label'] != 'Benign').astype(int)
        
        print("✅ İkili etiketler oluşturuldu:")
        print(f"   0: Benign (Normal trafik)")
        print(f"   1: Attack (Saldırı trafiği)")
        
        print(f"\nBenign: {(self.data['is_attack']==0).sum():,}, "
              f"Attack: {(self.data['is_attack']==1).sum():,}")
        
        return self
    
    def select_features(self):
        """Özellik seçimi yap"""
        print("\n" + "="*80)
        print("🔧 ÖZELLİK SEÇİMİ")
        print("="*80)
        
        # Label ve timestamp gibi sütunları çıkar
        exclude_cols = ['Label', 'is_attack', 'Timestamp']
        
        # Tüm sütunları al
        all_cols = self.data.columns.tolist()
        
        # Feature sütunlarını seç
        self.feature_columns = [col for col in all_cols if col not in exclude_cols]
        
        # Sadece sayısal sütunları tut
        numeric_features = []
        for col in self.feature_columns:
            if pd.api.types.is_numeric_dtype(self.data[col]):
                numeric_features.append(col)
        
        self.feature_columns = numeric_features
        
        print(f"✅ {len(self.feature_columns)} sayısal özellik seçildi")
        print(f"   İlk 10 özellik: {self.feature_columns[:10]}")
        
        return self
    
    def normalize_features(self):
        """Sayısal özellikleri normalize et"""
        print("\n" + "="*80)
        print("📏 ÖZELLİKLER NORMALİZE EDİLİYOR")
        print("="*80)
        
        # Scaler'ı fit ve transform et
        self.data[self.feature_columns] = self.scaler.fit_transform(
            self.data[self.feature_columns]
        )
        
        print(f"✅ {len(self.feature_columns)} özellik normalize edildi")
        print(f"   Ortalama: ~0, Standart sapma: ~1")
        
        return self
    
    def split_data(self, test_size=0.2, random_state=42):
        """Eğitim ve test setlerine ayır"""
        print("\n" + "="*80)
        print("✂️  VERİ AYIRMA")
        print("="*80)
        
        X = self.data[self.feature_columns].values
        y = self.data['is_attack'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\n✅ Eğitim seti:")
        print(f"   X_train shape: {X_train.shape}")
        print(f"   y_train shape: {y_train.shape}")
        print(f"   - Benign: {(y_train==0).sum():,}")
        print(f"   - Attack: {(y_train==1).sum():,}")
        
        print(f"\n✅ Test seti:")
        print(f"   X_test shape: {X_test.shape}")
        print(f"   y_test shape: {y_test.shape}")
        print(f"   - Benign: {(y_test==0).sum():,}")
        print(f"   - Attack: {(y_test==1).sum():,}")
        
        return X_train, y_train, X_test, y_test
    
    def save_processed_data(self, output_dir='processed_data_cicids'):
        """İşlenmiş veriyi kaydet"""
        print("\n" + "="*80)
        print("💾 VERİ KAYDEDİLİYOR")
        print("="*80)
        
        # Klasör oluştur
        os.makedirs(output_dir, exist_ok=True)
        
        # Veriyi ayır ve kaydet
        X_train, y_train, X_test, y_test = self.split_data()
        
        np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
        
        # Özellik isimlerini kaydet
        with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
            f.write('\n'.join(self.feature_columns))
        
        # Metadata kaydet
        metadata = {
            'dataset': 'CICIDS2018',
            'num_features': len(self.feature_columns),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_benign': (y_train==0).sum(),
            'train_attack': (y_train==1).sum(),
            'test_benign': (y_test==0).sum(),
            'test_attack': (y_test==1).sum(),
        }
        np.save(os.path.join(output_dir, 'metadata.npy'), metadata)
        
        print(f"✅ Veriler '{output_dir}' klasörüne kaydedildi:")
        print(f"   - X_train.npy")
        print(f"   - y_train.npy")
        print(f"   - X_test.npy")
        print(f"   - y_test.npy")
        print(f"   - feature_names.txt")
        print(f"   - metadata.npy")
        
        return X_train, y_train, X_test, y_test


def main():
    """Ana fonksiyon - Tüm veri ön işleme adımlarını çalıştır"""
    print("\n" + "="*80)
    print("🚀 CICIDS2018 - VERİ ÖN İŞLEME")
    print("="*80 + "\n")
    
    # Preprocessor oluştur
    preprocessor = CICIDS2018Preprocessor()
    
    # Adım adım veri hazırlama
    (preprocessor
     .load_data()
     .explore_data()
     .clean_data()
     .create_binary_labels()
     .select_features()
     .normalize_features())
    
    # Veriyi kaydet
    X_train, y_train, X_test, y_test = preprocessor.save_processed_data()
    
    # Özet
    print("\n" + "="*80)
    print("✨ CICIDS2018 VERİ ÖN İŞLEME TAMAMLANDI!")
    print("="*80)
    print(f"\n   Veriler 'processed_data_cicids' klasöründe kullanıma hazır!")
    
    return preprocessor, X_train, y_train, X_test, y_test


if __name__ == "__main__":
    preprocessor, X_train, y_train, X_test, y_test = main()
