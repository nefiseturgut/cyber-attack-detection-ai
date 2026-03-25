"""
LSTM Siber Saldırı Tespit Modeli
Ağ trafiğindeki zaman serisi pattern'lerini öğrenerek saldırı tespiti yapar
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime


class CyberAttackLSTM:
    """LSTM tabanlı siber saldırı tespit modeli"""
    
    def __init__(self, sequence_length, n_features):
        """
        Args:
            sequence_length: Sequence uzunluğu
            n_features: Özellik sayısı
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.history = None
        
    def build_model(self, lstm_units=[128, 64], dropout_rate=0.3, learning_rate=0.001):
        """
        LSTM modelini oluştur
        
        Args:
            lstm_units: Her LSTM katmanındaki birim sayısı listesi
            dropout_rate: Dropout oranı (overfitting'i önlemek için)
            learning_rate: Öğrenme oranı
        """
        print("\n" + "="*80)
        print("🏗️  LSTM MODELİ OLUŞTURULUYOR")
        print("="*80)
        
        model = keras.Sequential(name='CyberAttack_LSTM')
        
        # İlk LSTM katmanı
        model.add(layers.Input(shape=(self.sequence_length, self.n_features)))
        model.add(layers.LSTM(
            units=lstm_units[0],
            return_sequences=True if len(lstm_units) > 1 else False,
            name='lstm_1'
        ))
        model.add(layers.Dropout(dropout_rate, name='dropout_1'))
        
        # Ek LSTM katmanları
        for i, units in enumerate(lstm_units[1:], start=2):
            return_seq = i < len(lstm_units)
            model.add(layers.LSTM(
                units=units,
                return_sequences=return_seq,
                name=f'lstm_{i}'
            ))
            model.add(layers.Dropout(dropout_rate, name=f'dropout_{i}'))
        
        # Dense katmanlar
        model.add(layers.Dense(32, activation='relu', name='dense_1'))
        model.add(layers.Dropout(dropout_rate, name='dropout_final'))
        
        # Çıkış katmanı (Binary classification)
        model.add(layers.Dense(1, activation='sigmoid', name='output'))
        
        # Model derleme
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        self.model = model
        
        print("\n✅ Model oluşturuldu!")
        print(f"\n📋 Model Mimarisi:")
        model.summary()
        
        return self
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=256):
        """
        Modeli eğit
        
        Args:
            X_train: Eğitim verisi
            y_train: Eğitim etiketleri
            X_val: Validation verisi
            y_val: Validation etiketleri
            epochs: Epoch sayısı
            batch_size: Batch boyutu
        """
        print("\n" + "="*80)
        print("🎓 MODEL EĞİTİMİ BAŞLIYOR")
        print("="*80)
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
        
        # Model checkpoint
        os.makedirs('models', exist_ok=True)
        checkpoint_path = 'models/best_lstm_model.keras'
        model_checkpoint = callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        )
        
        # Class weights hesapla (dengesiz veri için)
        n_normal = (y_train == 0).sum()
        n_attack = (y_train == 1).sum()
        total = len(y_train)
        
        class_weight = {
            0: total / (2 * n_normal),
            1: total / (2 * n_attack)
        }
        
        print(f"\n⚖️  Class weights:")
        print(f"   Normal (0): {class_weight[0]:.3f}")
        print(f"   Saldırı (1): {class_weight[1]:.3f}")
        
        print(f"\n🏃 Eğitim başlıyor...")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Train samples: {len(X_train):,}")
        print(f"   Validation samples: {len(X_val):,}")
        
        # Eğit
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=[early_stopping, reduce_lr, model_checkpoint],
            verbose=1
        )
        
        print("\n✅ Eğitim tamamlandı!")
        print(f"💾 En iyi model kaydedildi: {checkpoint_path}")
        
        return self
    
    def evaluate(self, X_test, y_test):
        """
        Modeli değerlendir
        
        Args:
            X_test: Test verisi
            y_test: Test etiketleri
        """
        print("\n" + "="*80)
        print("📊 MODEL DEĞERLENDİRME")
        print("="*80)
        
        # Tahmin yap
        print("\n🔮 Tahminler yapılıyor...")
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Metrikler
        test_loss, test_acc, test_prec, test_rec, test_auc = self.model.evaluate(
            X_test, y_test, verbose=0
        )
        
        print(f"\n📈 Test Sonuçları:")
        print(f"   Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"   Precision: {test_prec:.4f}")
        print(f"   Recall:    {test_rec:.4f}")
        print(f"   F1-Score:  {2 * (test_prec * test_rec) / (test_prec + test_rec):.4f}")
        print(f"   AUC:       {test_auc:.4f}")
        print(f"   Loss:      {test_loss:.4f}")
        
        # Classification report
        print(f"\n📋 Detaylı Sınıflandırma Raporu:")
        print(classification_report(
            y_test, y_pred,
            target_names=['Normal', 'Saldırı'],
            digits=4
        ))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n🔢 Confusion Matrix:")
        print(f"                 Predicted")
        print(f"               Normal  Saldırı")
        print(f"   Normal      {cm[0][0]:6}  {cm[0][1]:6}")
        print(f"   Saldırı     {cm[1][0]:6}  {cm[1][1]:6}")
        
        # Doğru ve yanlış tahminler
        true_negatives = cm[0][0]
        false_positives = cm[0][1]
        false_negatives = cm[1][0]
        true_positives = cm[1][1]
        
        print(f"\n✅ Doğru Tahminler:")
        print(f"   True Negatives (Normal→Normal):   {true_negatives:,}")
        print(f"   True Positives (Saldırı→Saldırı): {true_positives:,}")
        
        print(f"\n❌ Yanlış Tahminler:")
        print(f"   False Positives (Normal→Saldırı): {false_positives:,}")
        print(f"   False Negatives (Saldırı→Normal): {false_negatives:,}")
        
        return {
            'accuracy': test_acc,
            'precision': test_prec,
            'recall': test_rec,
            'f1_score': 2 * (test_prec * test_rec) / (test_prec + test_rec),
            'auc': test_auc,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def plot_training_history(self, save_path='models/training_history.png'):
        """Eğitim geçmişini görselleştir"""
        if self.history is None:
            print("⚠️  Model henüz eğitilmedi!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('LSTM Model Eğitim Geçmişi', fontsize=16, fontweight='bold')
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train', linewidth=2)
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation', linewidth=2)
        axes[0, 0].set_title('Model Accuracy', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train', linewidth=2)
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation', linewidth=2)
        axes[0, 1].set_title('Model Loss', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Train', linewidth=2)
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation', linewidth=2)
        axes[1, 0].set_title('Model Precision', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # AUC
        axes[1, 1].plot(self.history.history['auc'], label='Train', linewidth=2)
        axes[1, 1].plot(self.history.history['val_auc'], label='Validation', linewidth=2)
        axes[1, 1].set_title('Model AUC', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Eğitim grafiği kaydedildi: {save_path}")
        
    def plot_confusion_matrix(self, cm, save_path='models/confusion_matrix.png'):
        """Confusion matrix görselleştir"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Saldırı'],
            yticklabels=['Normal', 'Saldırı'],
            cbar_kws={'label': 'Sayı'}
        )
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('Gerçek Etiket', fontsize=12)
        plt.xlabel('Tahmin Edilen Etiket', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Confusion matrix kaydedildi: {save_path}")


def main():
    """Ana fonksiyon - LSTM modeli eğitimi"""
    print("\n" + "="*80)
    print("🚀 LSTM SİBER SALDIRI TESPİT MODELİ")
    print("="*80 + "\n")
    
    # Veriyi yükle
    print("📂 LSTM verisi yükleniyor...")
    lstm_dir = 'lstm_data'
    
    X_train = np.load(os.path.join(lstm_dir, 'X_train_seq.npy'))
    y_train = np.load(os.path.join(lstm_dir, 'y_train_seq.npy'))
    X_test = np.load(os.path.join(lstm_dir, 'X_test_seq.npy'))
    y_test = np.load(os.path.join(lstm_dir, 'y_test_seq.npy'))
    metadata = np.load(os.path.join(lstm_dir, 'metadata.npy'), allow_pickle=True).item()
    
    print(f"✅ Veri yüklendi:")
    print(f"   Train: {X_train.shape}")
    print(f"   Test: {X_test.shape}")
    print(f"   Sequence length: {metadata['sequence_length']}")
    print(f"   Features: {metadata['n_features']}")
    
    # Validation split
    val_split = 0.2
    split_idx = int(len(X_train) * (1 - val_split))
    X_val = X_train[split_idx:]
    y_val = y_train[split_idx:]
    X_train = X_train[:split_idx]
    y_train = y_train[:split_idx]
    
    print(f"\n📊 Veri bölümü:")
    print(f"   Train: {X_train.shape[0]:,} samples")
    print(f"   Validation: {X_val.shape[0]:,} samples")
    print(f"   Test: {X_test.shape[0]:,} samples")
    
    # Model oluştur ve eğit
    lstm_model = CyberAttackLSTM(
        sequence_length=metadata['sequence_length'],
        n_features=metadata['n_features']
    )
    
    lstm_model.build_model(
        lstm_units=[128, 64],  # 2 LSTM katmanı
        dropout_rate=0.3,
        learning_rate=0.001
    )
    
    lstm_model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=256
    )
    
    # Değerlendir
    results = lstm_model.evaluate(X_test, y_test)
    
    # Grafikleri kaydet
    lstm_model.plot_training_history()
    lstm_model.plot_confusion_matrix(results['confusion_matrix'])
    
    print("\n" + "="*80)
    print("✨ LSTM MODELİ EĞİTİMİ TAMAMLANDI!")
    print("="*80)
    print(f"\n🎯 Final Test Accuracy: {results['accuracy']*100:.2f}%")
    print(f"📁 Model ve grafikler 'models/' klasöründe kaydedildi")
    print(f"\n🛡️  Model artık ağ trafiğini izleyip saldırı tespiti yapabilir!")
    
    return lstm_model, results


if __name__ == "__main__":
    lstm_model, results = main()
