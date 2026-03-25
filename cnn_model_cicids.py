# -*- coding: utf-8 -*-
"""
CNN Model - CICIDS2018 Dataset
Convolutional Neural Network ile Siber Saldiri Tespiti
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CNN MODEL - CICIDS2018 DATASET")
print("="*80)
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
print("="*80)

# Veriyi yukle
print("\n[*] CICIDS2018 verisi yukleniyor...")
X_train = np.load('lstm_data_cicids/X_train_seq.npy')
y_train = np.load('lstm_data_cicids/y_train_seq.npy')
X_test = np.load('lstm_data_cicids/X_test_seq.npy')
y_test = np.load('lstm_data_cicids/y_test_seq.npy')

print(f"  [OK] Train data: {X_train.shape}")
print(f"  [OK] Test data: {X_test.shape}")

# CNN icin reshape (samples, timesteps, features) -> (samples, timesteps, features, 1)
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

print(f"  [OK] Train CNN shape: {X_train_cnn.shape}")
print(f"  [OK] Test CNN shape: {X_test_cnn.shape}")

# Veri istatistikleri
unique, counts = np.unique(y_train, return_counts=True)
print(f"\n[*] Sinif dagilimi (Train):")
for label, count in zip(unique, counts):
    print(f"  Class {label}: {count:,} samples ({count/len(y_train)*100:.1f}%)")

# Model parametreleri
sequence_length = X_train.shape[1]  # 10
n_features = X_train.shape[2]  # 78
print(f"\n[*] Model parametreleri:")
print(f"  Sequence length: {sequence_length}")
print(f"  Number of features: {n_features}")

# CNN Model Olustur
print("\n[*] CNN modeli olusturuluyor...")

model = keras.Sequential([
    # Input layer
    layers.Input(shape=(sequence_length, n_features, 1)),
    
    # Conv1D Layer 1
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='conv1'),
    layers.BatchNormalization(name='bn1'),
    layers.MaxPooling2D(pool_size=(2, 2), name='pool1'),
    layers.Dropout(0.3, name='dropout1'),
    
    # Conv1D Layer 2
    layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name='conv2'),
    layers.BatchNormalization(name='bn2'),
    layers.MaxPooling2D(pool_size=(2, 2), name='pool2'),
    layers.Dropout(0.3, name='dropout2'),
    
    # Flatten
    layers.Flatten(name='flatten'),
    
    # Dense layers
    layers.Dense(128, activation='relu', name='dense1'),
    layers.Dropout(0.4, name='dropout3'),
    layers.Dense(64, activation='relu', name='dense2'),
    layers.Dropout(0.4, name='dropout4'),
    
    # Output layer
    layers.Dense(1, activation='sigmoid', name='output')
], name='CICIDS_CNN')

# Model ozeti
print("\n" + "="*80)
print("MODEL MIMARISI")
print("="*80)
model.summary()

# Class weights
class_weight = {
    0: len(y_train) / (2 * counts[0]),
    1: len(y_train) / (2 * counts[1])
}
print(f"\n[*] Class weights: {class_weight}")

# Model compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc')
    ]
)

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.00001,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'models/best_cnn_model_cicids.keras',
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    )
]

print("\n" + "="*80)
print("MODEL EGITIMI BASLIYOR")
print("="*80)
print(f"Baslangic: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Model egitimi
history = model.fit(
    X_train_cnn, y_train,
    validation_data=(X_test_cnn, y_test),
    epochs=30,
    batch_size=256,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)

print(f"\nBitis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# En iyi modeli yukle
print("\n[*] En iyi model yukleniyor...")
model = keras.models.load_model('models/best_cnn_model_cicids.keras')

# Test seti uzerinde degerlendirme
print("\n[*] Model degerlendiriliyor...")
y_pred_proba = model.predict(X_test_cnn, verbose=0).flatten()
y_pred = (y_pred_proba >= 0.5).astype(int)

# Metrikler
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "="*80)
print("CICIDS2018 - CNN MODEL PERFORMANSI")
print("="*80)
print(f"  Accuracy:  {accuracy*100:.2f}%")
print(f"  Precision: {precision*100:.2f}%")
print(f"  Recall:    {recall*100:.2f}%")
print(f"  F1-Score:  {f1*100:.2f}%")
print(f"  AUC:       {auc:.4f}")
print("="*80)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n[*] Confusion Matrix:")
print(cm)

# Gorselestirme 1: Training History
print("\n[*] Egitim grafikleri olusturuluyor...")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('CICIDS2018 - CNN Training History', fontsize=16, fontweight='bold')

# Accuracy
axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2)
axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
axes[0, 0].set_title('Accuracy', fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Loss
axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2)
axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2)
axes[0, 1].set_title('Loss', fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Precision
axes[1, 0].plot(history.history['precision'], label='Train', linewidth=2)
axes[1, 0].plot(history.history['val_precision'], label='Validation', linewidth=2)
axes[1, 0].set_title('Precision', fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# AUC
axes[1, 1].plot(history.history['auc'], label='Train', linewidth=2)
axes[1, 1].plot(history.history['val_auc'], label='Validation', linewidth=2)
axes[1, 1].set_title('AUC', fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('AUC')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('models/cnn_cicids_training_history.png', dpi=300, bbox_inches='tight')
print("  [OK] models/cnn_cicids_training_history.png")

# Gorselestirme 2: Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=['Normal', 'Saldiri'],
            yticklabels=['Normal', 'Saldiri'],
            cbar_kws={'label': 'Sample Count'})
plt.title('CICIDS2018 - CNN Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Gercek Sinif', fontsize=13, fontweight='bold')
plt.xlabel('Tahmin Edilen Sinif', fontsize=13, fontweight='bold')

tn, fp, fn, tp = cm.ravel()
total = tn + fp + fn + tp
stats_text = f"""
Toplam: {total:,} samples

True Negatives:  {tn:,} ({tn/total*100:.1f}%)
False Positives: {fp:,} ({fp/total*100:.1f}%)
False Negatives: {fn:,} ({fn/total*100:.1f}%)
True Positives:  {tp:,} ({tp/total*100:.1f}%)
"""
plt.gcf().text(0.02, 0.02, stats_text, fontsize=10,
               family='monospace',
               bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.5))

plt.tight_layout()
plt.savefig('models/cnn_cicids_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("  [OK] models/cnn_cicids_confusion_matrix.png")

# Sonuc raporu
report_text = f"""
{'='*80}
CICIDS2018 - CNN MODEL RAPORU
{'='*80}

Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

1. DATASET BILGILERI
{'='*80}
Dataset: CICIDS2018
Train samples: {len(y_train):,}
Test samples: {len(y_test):,}
Features: {n_features}
Sequence length: {sequence_length}

Sinif dagilimi:
  Normal (0): {counts[0]:,} ({counts[0]/len(y_train)*100:.1f}%)
  Saldiri (1): {counts[1]:,} ({counts[1]/len(y_train)*100:.1f}%)

2. MODEL MIMARISI
{'='*80}
- Conv2D Layer 1: 64 filters, kernel=(3,3), ReLU
- BatchNormalization
- MaxPooling2D: (2,2)
- Dropout: 0.3

- Conv2D Layer 2: 128 filters, kernel=(3,3), ReLU
- BatchNormalization
- MaxPooling2D: (2,2)
- Dropout: 0.3

- Flatten
- Dense: 128 units (ReLU)
- Dropout: 0.4
- Dense: 64 units (ReLU)
- Dropout: 0.4
- Output: 1 unit (Sigmoid)

Total parameters: ~{model.count_params():,}

3. EGITIM PARAMETRELERI
{'='*80}
- Optimizer: Adam (lr=0.001)
- Loss: Binary Crossentropy
- Epochs: {len(history.history['loss'])}
- Batch size: 256
- Class weights: Kullanildi
- Early stopping: Patience=5
- Learning rate reduction: Factor=0.5, Patience=3

4. PERFORMANS METRIKLERI
{'='*80}
Test Seti Sonuclari:
  * Accuracy:  {accuracy*100:.2f}%
  * Precision: {precision*100:.2f}%
  * Recall:    {recall*100:.2f}%
  * F1-Score:  {f1*100:.2f}%
  * AUC:       {auc:.4f}

Confusion Matrix:
  True Negatives:  {tn:,}
  False Positives: {fp:,}
  False Negatives: {fn:,}
  True Positives:  {tp:,}

5. SONUC
{'='*80}
CICIDS2018 dataseti uzerinde CNN modeli basariyla egitildi.
Model, {accuracy*100:.2f}% dogruluk orani ile siber saldirilari tespit edebiliyor.

CNN'in Avantajlari:
  - Spatial pattern detection
  - Feature extraction otomatik
  - Convolution ile lokal korelasyonlar
  - Pooling ile boyut azaltma

Kaydedilen dosyalar:
  - models/best_cnn_model_cicids.keras
  - models/cnn_cicids_training_history.png
  - models/cnn_cicids_confusion_matrix.png

{'='*80}
Gelistirici: Nefise
Tarih: {datetime.now().strftime('%d %B %Y')}
{'='*80}
"""

with open('models/cnn_cicids_report.txt', 'w', encoding='utf-8') as f:
    f.write(report_text)

print("\n[OK] models/cnn_cicids_report.txt")

print("\n" + "="*80)
print("[SUCCESS] CICIDS2018 - CNN MODEL TAMAMLANDI!")
print("="*80)
print("\nOlusturulan dosyalar:")
print("  [OK] models/best_cnn_model_cicids.keras")
print("  [OK] models/cnn_cicids_training_history.png")
print("  [OK] models/cnn_cicids_confusion_matrix.png")
print("  [OK] models/cnn_cicids_report.txt")
print("\n" + "="*80)
