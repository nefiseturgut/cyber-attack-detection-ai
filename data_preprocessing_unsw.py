"""
Siber Saldırı Tespit Projesi - UNSW_NB15 Veri Ön İşleme
UNSW_NB15 veri seti için veri hazırlama scripti
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path


class UNSWNB15Preprocessor:
    """UNSW_NB15 verisi için ön işleme sınıfı"""
    
    def __init__(self, data_dir='datasets/UNSW_NB15'):
        self.data_dir = Path(data_dir)
        self.train_data = None
        self.test_data = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def load_data(self):
        """Veri setlerini yükle"""
        print("📂 UNSW_NB15 verisi yükleniyor...")
        
        train_path = self.data_dir / 'UNSW_NB15_training-set.csv'
        test_path = self.data_dir / 'UNSW_NB15_testing-set.csv'
        
        self.train_data = pd.read_csv(train_path, low_memory=False)
        self.test_data = pd.read_csv(test_path, low_memory=False)
        
        print(f"✅ Train verisi yüklendi: {self.train_data.shape}")
        print(f"✅ Test verisi yüklendi: {self.test_data.shape}")
        
        return self
    
    def explore_data(self):
        """Veriyi keşfet ve temel istatistikleri göster"""
        print("\n" + "="*80)
        print("📊 VERİ KEŞFİ - UNSW_NB15")
        print("="*80)
        
        print(f"\nTrain - Satır sayısı: {self.train_data.shape[0]:,}")
        print(f"Train - Sütun sayısı: {self.train_data.shape[1]}")
        print(f"\nTest - Satır sayısı: {self.test_data.shape[0]:,}")
        print(f"Test - Sütun sayısı: {self.test_data.shape[1]}")
        
        print("\n--- İlk 10 Sütun ---")
        print(self.train_data.columns[:10].tolist())
        
        print("\n--- Veri Tipleri ---")
        print(self.train_data.dtypes.value_counts())
        
        print("\n--- Eksik Değerler (Train) ---")
        missing = self.train_data.isnull().sum()
        if missing.sum() > 0:
            print(f"Toplam eksik değer: {missing.sum():,}")
            print(missing[missing > 0].head(10))
        else:
            print("✅ Eksik değer yok!")
        
        # Binary label kontrolü
        if 'label' in self.train_data.columns:
            print("\n--- Binary Label Dağılımı (Train) ---")
            label_counts = self.train_data['label'].value_counts()
            print(label_counts)
            
            normal_count = (self.train_data['label'] == 0).sum()
            attack_count = (self.train_data['label'] == 1).sum()
            print(f"\nNormal: {normal_count:,} ({normal_count/len(self.train_data)*100:.1f}%)")
            print(f"Attack: {attack_count:,} ({attack_count/len(self.train_data)*100:.1f}%)")
        
        # Attack category kontrolü
        if 'attack_cat' in self.train_data.columns:
            print("\n--- Attack Category Dağılımı (Train) ---")
            attack_cats = self.train_data['attack_cat'].value_counts()
            print(attack_cats.head(15))
        
        return self
    
    def clean_data(self):
        """Veriyi temizle"""
        print("\n" + "="*80)
        print("🧹 VERİ TEMİZLEME")
        print("="*80)
        
        train_initial = self.train_data.shape
        test_initial = self.test_data.shape
        
        # Infinity ve NaN değerlerini temizle
        print("\n🔍 Infinity ve NaN değerleri kontrol ediliyor...")
        self.train_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.test_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # NaN satırları kaldır
        train_nan = self.train_data.isnull().sum().sum()
        test_nan = self.test_data.isnull().sum().sum()
        
        if train_nan > 0:
            self.train_data.dropna(inplace=True)
            print(f"✅ Train: {train_nan:,} NaN içeren satır kaldırıldı")
        
        if test_nan > 0:
            self.test_data.dropna(inplace=True)
            print(f"✅ Test: {test_nan:,} NaN içeren satır kaldırıldı")
        
        # Duplikatları kontrol et
        train_dup = self.train_data.duplicated().sum()
        test_dup = self.test_data.duplicated().sum()
        
        print(f"\nTrain duplikat: {train_dup:,}")
        print(f"Test duplikat: {test_dup:,}")
        
        if train_dup > 0:
            self.train_data = self.train_data.drop_duplicates()
            print(f"✅ Train duplikatlar kaldırıldı")
        
        if test_dup > 0:
            self.test_data = self.test_data.drop_duplicates()
            print(f"✅ Test duplikatlar kaldırıldı")
        
        print(f"\nTrain boyutu: {train_initial} → {self.train_data.shape}")
        print(f"Test boyutu: {test_initial} → {self.test_data.shape}")
        
        return self
    
    def prepare_labels(self):
        """Etiketleri hazırla"""
        print("\n" + "="*80)
        print("🏷️  ETİKET HAZIRLIĞI")
        print("="*80)
        
        if 'label' not in self.train_data.columns:
            raise ValueError("'label' sütunu bulunamadı!")
        
        # UNSW_NB15'te label zaten binary (0/1)
        # is_attack olarak kopyala
        self.train_data['is_attack'] = self.train_data['label']
        self.test_data['is_attack'] = self.test_data['label']
        
        print("✅ Binary etiketler hazır:")
        print(f"   0: Normal trafik")
        print(f"   1: Attack (Saldırı trafiği)")
        
        print(f"\nTrain - Normal: {(self.train_data['is_attack']==0).sum():,}, "
              f"Attack: {(self.train_data['is_attack']==1).sum():,}")
        print(f"Test - Normal: {(self.test_data['is_attack']==0).sum():,}, "
              f"Attack: {(self.test_data['is_attack']==1).sum():,}")
        
        return self
    
    def select_features(self):
        """Özellik seçimi yap"""
        print("\n" + "="*80)
        print("🔧 ÖZELLİK SEÇİMİ")
        print("="*80)
        
        # Exclude sütunları
        exclude_cols = ['id', 'label', 'is_attack', 'attack_cat']
        
        # Kategorik sütunlar (eğer varsa encode edilecek)
        categorical_cols = ['proto', 'service', 'state']
        
        # Tüm sütunları al
        all_cols = self.train_data.columns.tolist()
        
        # Feature sütunlarını seç
        feature_cols = [col for col in all_cols if col not in exclude_cols]
        
        # Kategorik sütunları encode et
        for col in categorical_cols:
            if col in feature_cols:
                print(f"   Encoding: {col}")
                le = LabelEncoder()
                
                # Birleştir, fit et, transform et
                combined = pd.concat([
                    self.train_data[col].astype(str),
                    self.test_data[col].astype(str)
                ]).unique()
                
                le.fit(combined)
                
                self.train_data[col] = le.transform(self.train_data[col].astype(str))
                self.test_data[col] = le.transform(self.test_data[col].astype(str))
                
                self.label_encoders[col] = le
        
        # Sadece sayısal sütunları tut
        numeric_features = []
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(self.train_data[col]):
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
        
        # Train ile scaler'ı fit et
        self.scaler.fit(self.train_data[self.feature_columns])
        
        # Her iki seti de transform et
        self.train_data[self.feature_columns] = self.scaler.transform(
            self.train_data[self.feature_columns]
        )
        self.test_data[self.feature_columns] = self.scaler.transform(
            self.test_data[self.feature_columns]
        )
        
        print(f"✅ {len(self.feature_columns)} özellik normalize edildi")
        print(f"   Ortalama: ~0, Standart sapma: ~1")
        
        return self
    
    def prepare_for_training(self):
        """Eğitim için final veri setlerini hazırla"""
        print("\n" + "="*80)
        print("🎯 EĞİTİM VERİLERİ HAZIRLANIYOR")
        print("="*80)
        
        # X (özellikler) ve y (etiketler) ayır
        X_train = self.train_data[self.feature_columns].values
        y_train = self.train_data['is_attack'].values
        
        X_test = self.test_data[self.feature_columns].values
        y_test = self.test_data['is_attack'].values
        
        print(f"\n✅ Eğitim seti hazır:")
        print(f"   X_train shape: {X_train.shape}")
        print(f"   y_train shape: {y_train.shape}")
        print(f"   - Normal: {(y_train==0).sum():,}")
        print(f"   - Attack: {(y_train==1).sum():,}")
        
        print(f"\n✅ Test seti hazır:")
        print(f"   X_test shape: {X_test.shape}")
        print(f"   y_test shape: {y_test.shape}")
        print(f"   - Normal: {(y_test==0).sum():,}")
        print(f"   - Attack: {(y_test==1).sum():,}")
        
        return X_train, y_train, X_test, y_test
    
    def save_processed_data(self, output_dir='processed_data_unsw'):
        """İşlenmiş veriyi kaydet"""
        print("\n" + "="*80)
        print("💾 VERİ KAYDEDİLİYOR")
        print("="*80)
        
        # Klasör oluştur
        os.makedirs(output_dir, exist_ok=True)
        
        # Veriyi hazırla
        X_train, y_train, X_test, y_test = self.prepare_for_training()
        
        # Kaydet
        np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
        
        # Özellik isimlerini kaydet
        with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
            f.write('\n'.join(self.feature_columns))
        
        # Metadata kaydet
        metadata = {
            'dataset': 'UNSW_NB15',
            'num_features': len(self.feature_columns),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_normal': (y_train==0).sum(),
            'train_attack': (y_train==1).sum(),
            'test_normal': (y_test==0).sum(),
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
    print("🚀 UNSW_NB15 - VERİ ÖN İŞLEME")
    print("="*80 + "\n")
    
    # Preprocessor oluştur
    preprocessor = UNSWNB15Preprocessor()
    
    # Adım adım veri hazırlama
    (preprocessor
     .load_data()
     .explore_data()
     .clean_data()
     .prepare_labels()
     .select_features()
     .normalize_features())
    
    # Veriyi kaydet
    X_train, y_train, X_test, y_test = preprocessor.save_processed_data()
    
    # Özet
    print("\n" + "="*80)
    print("✨ UNSW_NB15 VERİ ÖN İŞLEME TAMAMLANDI!")
    print("="*80)
    print(f"\n   Veriler 'processed_data_unsw' klasöründe kullanıma hazır!")
    
    return preprocessor, X_train, y_train, X_test, y_test


if __name__ == "__main__":
    preprocessor, X_train, y_train, X_test, y_test = main()
