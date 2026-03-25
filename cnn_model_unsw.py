# -*- coding: utf-8 -*-
"""
CNN Model - UNSW-NB15 Dataset
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CNN MODEL - UNSW-NB15 DATASET")
print("="*80)

X_train = np.load('lstm_data_unsw/X_train_seq.npy')
y_train = np.load('lstm_data_unsw/y_train_seq.npy')
X_test = np.load('lstm_data_unsw/X_test_seq.npy')
y_test = np.load('lstm_data_unsw/y_test_seq.npy')

print(f"  [OK] Train: {X_train.shape}, Test: {X_test.shape}")

# CNN reshape
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

unique, counts = np.unique(y_train, return_counts=True)
class_weight = {0: len(y_train) / (2 * counts[0]), 1: len(y_train) / (2 * counts[1])}

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1], X_train.shape[2], 1)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid')
], name='UNSW_CNN')

model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]
)

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1),
    keras.callbacks.ModelCheckpoint('models/best_cnn_model_unsw.keras', monitor='val_auc', mode='max', save_best_only=True, verbose=1)
]

print("\n" + "="*80)
print("EGITIM BASLIYOR")
print("="*80)

history = model.fit(
    X_train_cnn, y_train,
    validation_data=(X_test_cnn, y_test),
    epochs=30,
    batch_size=256,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)

model = keras.models.load_model('models/best_cnn_model_unsw.keras')
y_pred_proba = model.predict(X_test_cnn, verbose=0).flatten()
y_pred = (y_pred_proba >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "="*80)
print("UNSW-NB15 - CNN PERFORMANS")
print("="*80)
print(f"  Accuracy:  {acc*100:.2f}%")
print(f"  Precision: {prec*100:.2f}%")
print(f"  Recall:    {rec*100:.2f}%")
print(f"  F1-Score:  {f1*100:.2f}%")
print(f"  AUC:       {auc:.4f}")
print("="*80)

# Gorselestirme
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('UNSW-NB15 - CNN Training History')

axes[0, 0].plot(history.history['accuracy'], label='Train')
axes[0, 0].plot(history.history['val_accuracy'], label='Val')
axes[0, 0].set_title('Accuracy'); axes[0, 0].legend(); axes[0, 0].grid(True)

axes[0, 1].plot(history.history['loss'], label='Train')
axes[0, 1].plot(history.history['val_loss'], label='Val')
axes[0, 1].set_title('Loss'); axes[0, 1].legend(); axes[0, 1].grid(True)

axes[1, 0].plot(history.history['precision'], label='Train')
axes[1, 0].plot(history.history['val_precision'], label='Val')
axes[1, 0].set_title('Precision'); axes[1, 0].legend(); axes[1, 0].grid(True)

axes[1, 1].plot(history.history['auc'], label='Train')
axes[1, 1].plot(history.history['val_auc'], label='Val')
axes[1, 1].set_title('AUC'); axes[1, 1].legend(); axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('models/cnn_unsw_training_history.png', dpi=300)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=['Normal', 'Saldiri'], yticklabels=['Normal', 'Saldiri'])
plt.title('UNSW-NB15 - CNN Confusion Matrix')
plt.tight_layout()
plt.savefig('models/cnn_unsw_confusion_matrix.png', dpi=300)

print("\n[SUCCESS] UNSW-NB15 CNN tamamlandi!")
