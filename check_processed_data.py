import numpy as np
from pathlib import Path

print("="*80)
print("📊 PREPROCESSING SONUÇLARI KONTROLÜ")
print("="*80)

datasets = [
    ('processed_data', 'KDD Cup 1999'),
    ('processed_data_cicids', 'CICIDS2018'),
    ('processed_data_unsw', 'UNSW_NB15'),
]

for folder, name in datasets:
    path = Path(folder)
    if not path.exists():
        print(f"\n❌ {name}: Klasör bulunamadı ({folder})")
        continue
    
    print(f"\n✅ {name}:")
    print(f"   Klasör: {folder}")
    
    # Metadata kontrol
    meta_file = path / 'metadata.npy'
    if meta_file.exists():
        meta = np.load(meta_file, allow_pickle=True).item()
        print(f"   Özellik sayısı: {meta.get('num_features', 'N/A')}")
        print(f"   Train örnekleri: {meta.get('train_samples', 'N/A'):,}")
        print(f"   Test örnekleri: {meta.get('test_samples', 'N/A'):,}")
        print(f"   Train Normal: {meta.get('train_normal', meta.get('train_benign', 'N/A')):,}")
        print(f"   Train Attack: {meta.get('train_attack', 'N/A'):,}")
    else:
        # Dosyaları direkt oku
        X_train = np.load(path / 'X_train.npy')
        y_train = np.load(path / 'y_train.npy')
        X_test = np.load(path / 'X_test.npy')
        y_test = np.load(path / 'y_test.npy')
        
        print(f"   X_train shape: {X_train.shape}")
        print(f"   y_train shape: {y_train.shape}")
        print(f"   X_test shape: {X_test.shape}")
        print(f"   y_test shape: {y_test.shape}")

print("\n" + "="*80)
print("✨ KONTROL TAMAMLANDI!")
print("="*80)
