"""
KDD Cup 1999 için LSTM Sequence Veri Hazırlama
"""

import numpy as np
import os
from pathlib import Path

def create_sequences(X, y, sequence_length=10):
    """Create sequences for LSTM from flat data"""
    X_seq = []
    y_seq = []
    
    for i in range(len(X) - sequence_length + 1):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length - 1])
    
    return np.array(X_seq), np.array(y_seq)

def main():
    print("="*80)
    print("🔄 KDD Cup 1999 - LSTM Veri Hazırlama")
    print("="*80)
    
    # Load processed data
    data_dir = Path('processed_data')
    X_train = np.load(data_dir / 'X_train.npy')
    y_train = np.load(data_dir / 'y_train.npy')
    X_test = np.load(data_dir / 'X_test.npy')
    y_test = np.load(data_dir / 'y_test.npy')
    
    print(f"\n📂 Veri yüklendi:")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_test: {X_test.shape}")
    
    # Create sequences
    sequence_length = 10
    print(f"\n🔧 Sequence oluşturuluyor (length={sequence_length})...")
    
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)
    
    print(f"\n✅ Sequence'ler oluşturuldu:")
    print(f"   X_train_seq: {X_train_seq.shape}")
    print(f"   y_train_seq: {y_train_seq.shape}")
    print(f"   X_test_seq: {X_test_seq.shape}")
    print(f"   y_test_seq: {y_test_seq.shape}")
    
    # Save
    output_dir = Path('lstm_data_kdd')
    output_dir.mkdir(exist_ok=True)
    
    np.save(output_dir / 'X_train_seq.npy', X_train_seq)
    np.save(output_dir / 'y_train_seq.npy', y_train_seq)
    np.save(output_dir / 'X_test_seq.npy', X_test_seq)
    np.save(output_dir / 'y_test_seq.npy', y_test_seq)
    
    metadata = {
        'sequence_length': sequence_length,
        'num_features': X_train.shape[1],
        'train_samples': len(X_train_seq),
        'test_samples': len(X_test_seq),
    }
    np.save(output_dir / 'metadata.npy', metadata)
    
    print(f"\n💾 Kaydedildi: {output_dir}/")
    print("✨ KDD LSTM verisi hazır!")

if __name__ == "__main__":
    main()
