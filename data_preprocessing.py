"""
Siber Saldırı Tespit Projesi - Veri Ön İşleme
KDD Cup 1999 veri seti için veri hazırlama scripti
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os

# KDD Cup 1999 veri seti için sütun isimleri
COLUMN_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]


class CyberAttackDataPreprocessor:
    """Siber saldırı verisi için ön işleme sınıfı"""
    
    def __init__(self, data_dir='datasets'):
        self.data_dir = data_dir
        self.train_data = None
        self.test_data = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def load_data(self):
        """Veri setlerini yükle"""
        print("📂 Veri setleri yükleniyor...")
        
        # Train verisi
        train_path = os.path.join(self.data_dir, 'KDDTrain+.txt')
        self.train_data = pd.read_csv(train_path, header=None, names=COLUMN_NAMES)
        print(f"✅ Train verisi yüklendi: {self.train_data.shape}")
        
        # Test verisi
        test_path = os.path.join(self.data_dir, 'KDDTest+.txt')
        self.test_data = pd.read_csv(test_path, header=None, names=COLUMN_NAMES)
        print(f"✅ Test verisi yüklendi: {self.test_data.shape}")
        
        return self
    
    def explore_data(self):
        """Veriyi keşfet ve temel istatistikleri göster"""
        print("\n" + "="*80)
        print("📊 VERİ KEŞFİ")
        print("="*80)
        
        print("\n--- Train Verisi Bilgileri ---")
        print(f"Satır sayısı: {self.train_data.shape[0]:,}")
        print(f"Sütun sayısı: {self.train_data.shape[1]}")
        
        print("\n--- İlk 5 Satır ---")
        print(self.train_data.head())
        
        print("\n--- Veri Tipleri ---")
        print(self.train_data.dtypes.value_counts())
        
        print("\n--- Eksik Değerler ---")
        missing = self.train_data.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("✅ Eksik değer yok!")
        
        print("\n--- Saldırı Türleri Dağılımı ---")
        label_counts = self.train_data['label'].value_counts()
        print(label_counts.head(10))
        print(f"\nToplam farklı saldırı türü: {len(label_counts)}")
        
        # Normal vs Saldırı
        normal_count = (self.train_data['label'] == 'normal').sum()
        attack_count = (self.train_data['label'] != 'normal').sum()
        print(f"\nNormal trafik: {normal_count:,} ({normal_count/len(self.train_data)*100:.1f}%)")
        print(f"Saldırı trafiği: {attack_count:,} ({attack_count/len(self.train_data)*100:.1f}%)")
        
        return self
    
    def clean_data(self):
        """Veriyi temizle"""
        print("\n" + "="*80)
        print("🧹 VERİ TEMİZLEME")
        print("="*80)
        
        # Difficulty sütununu kaldır (KDD+ veri setinde kullanılmaz)
        if 'difficulty' in self.train_data.columns:
            self.train_data = self.train_data.drop('difficulty', axis=1)
            self.test_data = self.test_data.drop('difficulty', axis=1)
            print("✅ 'difficulty' sütunu kaldırıldı")
        
        # Duplikatları kontrol et
        train_duplicates = self.train_data.duplicated().sum()
        test_duplicates = self.test_data.duplicated().sum()
        print(f"\nTrain duplikat sayısı: {train_duplicates:,}")
        print(f"Test duplikat sayısı: {test_duplicates:,}")
        
        if train_duplicates > 0:
            self.train_data = self.train_data.drop_duplicates()
            print(f"✅ Train duplikatları kaldırıldı: {self.train_data.shape}")
        
        if test_duplicates > 0:
            self.test_data = self.test_data.drop_duplicates()
            print(f"✅ Test duplikatları kaldırıldı: {self.test_data.shape}")
        
        return self
    
    def create_binary_labels(self):
        """İkili sınıflandırma için etiketler oluştur (Normal vs Saldırı)"""
        print("\n" + "="*80)
        print("🏷️  ETİKET DÖNÜŞÜMÜ")
        print("="*80)
        
        # Binary sınıflandırma için yeni sütun oluştur
        self.train_data['is_attack'] = (self.train_data['label'] != 'normal').astype(int)
        self.test_data['is_attack'] = (self.test_data['label'] != 'normal').astype(int)
        
        print("✅ İkili etiketler oluşturuldu:")
        print(f"   0: Normal trafik")
        print(f"   1: Saldırı trafiği")
        
        print(f"\nTrain - Normal: {(self.train_data['is_attack']==0).sum():,}, "
              f"Saldırı: {(self.train_data['is_attack']==1).sum():,}")
        print(f"Test - Normal: {(self.test_data['is_attack']==0).sum():,}, "
              f"Saldırı: {(self.test_data['is_attack']==1).sum():,}")
        
        return self
    
    def encode_categorical_features(self):
        """Kategorik özellikleri sayısal değerlere dönüştür"""
        print("\n" + "="*80)
        print("🔢 KATEGORİK ÖZELLİKLER KODLANIYOR")
        print("="*80)
        
        # Kategorik sütunlar
        categorical_columns = ['protocol_type', 'service', 'flag']
        
        for col in categorical_columns:
            # Label Encoder oluştur ve train verisi ile fit et
            le = LabelEncoder()
            
            # Train ve test verilerini birleştir ve fit et
            combined_values = pd.concat([
                self.train_data[col], 
                self.test_data[col]
            ]).unique()
            
            le.fit(combined_values)
            
            # Transform işlemi
            self.train_data[col] = le.transform(self.train_data[col])
            self.test_data[col] = le.transform(self.test_data[col])
            
            self.label_encoders[col] = le
            
            print(f"✅ {col}: {len(le.classes_)} benzersiz değer kodlandı")
        
        return self
    
    def normalize_features(self):
        """Sayısal özellikleri normalize et"""
        print("\n" + "="*80)
        print("📏 ÖZELLİKLER NORMALİZE EDİLİYOR")
        print("="*80)
        
        # Özellik sütunları (label ve is_attack hariç)
        self.feature_columns = [col for col in self.train_data.columns 
                               if col not in ['label', 'is_attack']]
        
        # Train verisi ile scaler'ı fit et
        self.scaler.fit(self.train_data[self.feature_columns])
        
        # Her iki veri setini de transform et
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
        print(f"   - Saldırı: {(y_train==1).sum():,}")
        
        print(f"\n✅ Test seti hazır:")
        print(f"   X_test shape: {X_test.shape}")
        print(f"   y_test shape: {y_test.shape}")
        print(f"   - Normal: {(y_test==0).sum():,}")
        print(f"   - Saldırı: {(y_test==1).sum():,}")
        
        return X_train, y_train, X_test, y_test
    
    def save_processed_data(self, output_dir='processed_data'):
        """İşlenmiş veriyi kaydet"""
        print("\n" + "="*80)
        print("💾 VERİ KAYDEDİLİYOR")
        print("="*80)
        
        # Klasör oluştur
        os.makedirs(output_dir, exist_ok=True)
        
        # Veriyi kaydet
        X_train, y_train, X_test, y_test = self.prepare_for_training()
        
        np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
        
        # Özellik isimlerini kaydet
        with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
            f.write('\n'.join(self.feature_columns))
        
        print(f"✅ Veriler '{output_dir}' klasörüne kaydedildi:")
        print(f"   - X_train.npy")
        print(f"   - y_train.npy")
        print(f"   - X_test.npy")
        print(f"   - y_test.npy")
        print(f"   - feature_names.txt")
        
        return X_train, y_train, X_test, y_test
    
    def get_data_summary(self):
        """Veri özeti oluştur"""
        summary = {
            'train_samples': len(self.train_data),
            'test_samples': len(self.test_data),
            'num_features': len(self.feature_columns) if self.feature_columns else 0,
            'train_normal': (self.train_data['is_attack']==0).sum(),
            'train_attack': (self.train_data['is_attack']==1).sum(),
            'test_normal': (self.test_data['is_attack']==0).sum(),
            'test_attack': (self.test_data['is_attack']==1).sum(),
        }
        return summary


def main():
    """Ana fonksiyon - Tüm veri ön işleme adımlarını çalıştır"""
    print("\n" + "="*80)
    print("🚀 SİBER SALDIRI TESPİT PROJESİ - VERİ ÖN İŞLEME")
    print("="*80 + "\n")
    
    # Preprocessor oluştur
    preprocessor = CyberAttackDataPreprocessor()
    
    # Adım adım veri hazırlama
    (preprocessor
     .load_data()
     .explore_data()
     .clean_data()
     .create_binary_labels()
     .encode_categorical_features()
     .normalize_features())
    
    # Veriyi kaydet
    X_train, y_train, X_test, y_test = preprocessor.save_processed_data()
    
    # Özet
    print("\n" + "="*80)
    print("✨ VERİ ÖN İŞLEME TAMAMLANDI!")
    print("="*80)
    
    summary = preprocessor.get_data_summary()
    print(f"\n📊 Özet:")
    print(f"   Toplam özellik sayısı: {summary['num_features']}")
    print(f"   Train örnekleri: {summary['train_samples']:,}")
    print(f"   Test örnekleri: {summary['test_samples']:,}")
    print(f"\n   Veriler 'processed_data' klasöründe kullanıma hazır!")
    
    return preprocessor, X_train, y_train, X_test, y_test


if __name__ == "__main__":
    preprocessor, X_train, y_train, X_test, y_test = main()
