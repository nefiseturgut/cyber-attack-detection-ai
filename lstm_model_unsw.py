# -*- coding: utf-8 -*-
"""
LSTM Model - UNSW-NB15 Dataset
Siber Saldiri Tespiti icin LSTM Derin Ogrenme Modeli
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("LSTM MODEL - UNSW-NB15 DATASET")
print("="*80)

# Veriyi yukle
print("\n[*] UNSW-NB15 verisi yukleniyor...")
X_train = np.load('lstm_data_unsw/X_train_seq.npy')
y_train = np.load('lstm_data_unsw/y_train_seq.npy')
X_test = np.load('lstm_data_unsw/X_test_seq.npy')
y_test = np.load('lstm_data_unsw/y_test_seq.npy')

print(f"  [OK] Train: {X_train.shape}, Test: {X_test.shape}")

# Sinif dagilimi
unique, counts = np.unique(y_train, return_counts=True)
print(f"\n[*] Sinif dagilimi:")
for label, count in zip(unique, counts):
    print(f"  Class {label}: {count:,} ({count/len(y_train)*100:.1f}%)")

# Model
sequence_length = X_train.shape[1]
n_features = X_train.shape[2]

model = keras.Sequential([
    layers.Input(shape=(sequence_length, n_features)),
    layers.LSTM(128, return_sequences=True),
    layers.Dropout(0.3),
    layers.LSTM(64),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
], name='UNSW_LSTM')

model.summary()

# Class weights
class_weight = {
    0: len(y_train) / (2 * counts[0]),
    1: len(y_train) / (2 * counts[1])
}

model.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]
)

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1),
    keras.callbacks.ModelCheckpoint('models/best_lstm_model_unsw.keras', monitor='val_auc', mode='max', save_best_only=True, verbose=1)
]

print("\n" + "="*80)
print("EGITIM BASLIYOR - UNSW-NB15")
print("="*80)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=256,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)

# Degerlendirme
model = keras.models.load_model('models/best_lstm_model_unsw.keras')
y_pred_proba = model.predict(X_test, verbose=0).flatten()
y_pred = (y_pred_proba >= 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "="*80)
print("UNSW-NB15 - LSTM PERFORMANS")
print("="*80)
print(f"  Accuracy:  {accuracy*100:.2f}%")
print(f"  Precision: {precision*100:.2f}%")
print(f"  Recall:    {recall*100:.2f}%")
print(f"  F1-Score:  {f1*100:.2f}%")
print(f"  AUC:       {auc:.4f}")
print("="*80)

# Gorselestirme
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('UNSW-NB15 - LSTM Training History', fontsize=16, fontweight='bold')

axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2)
axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
axes[0, 0].set_title('Accuracy'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2)
axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2)
axes[0, 1].set_title('Loss'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(history.history['precision'], label='Train', linewidth=2)
axes[1, 0].plot(history.history['val_precision'], label='Validation', linewidth=2)
axes[1, 0].set_title('Precision'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(history.history['auc'], label='Train', linewidth=2)
axes[1, 1].plot(history.history['val_auc'], label='Validation', linewidth=2)
axes[1, 1].set_title('AUC'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('models/lstm_unsw_training_history.png', dpi=300, bbox_inches='tight')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Saldiri'], yticklabels=['Normal', 'Saldiri'])
plt.title('UNSW-NB15 - LSTM Confusion Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('models/lstm_unsw_confusion_matrix.png', dpi=300, bbox_inches='tight')

print("\n[SUCCESS] UNSW-NB15 LSTM tamamlandi!")
print("  [OK] models/best_lstm_model_unsw.keras")
print("  [OK] models/lstm_unsw_training_history.png")
print("  [OK] models/lstm_unsw_confusion_matrix.png")
