# -*- coding: utf-8 -*-
"""
CICIDS2018 - Overfitting Analizi
Tüm modellerin train/test performanslarını karşılaştırarak overfitting tespiti
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow import keras
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("*** CICIDS2018 - OVERFITTING ANALİZİ ***")
print("="*80)

# Veriyi yükle
print("\n[*] Veriler yukleniyor...")
X_train = np.load('processed_data_cicids/X_train.npy')
y_train = np.load('processed_data_cicids/y_train.npy')
X_test = np.load('processed_data_cicids/X_test.npy')
y_test = np.load('processed_data_cicids/y_test.npy')

X_train_seq = np.load('lstm_data_cicids/X_train_seq.npy')
y_train_seq = np.load('lstm_data_cicids/y_train_seq.npy')
X_test_seq = np.load('lstm_data_cicids/X_test_seq.npy')
y_test_seq = np.load('lstm_data_cicids/y_test_seq.npy')

# Sequence uzunluğuna göre hizala
seq_len = len(X_test_seq)
X_train_tabular = X_train[:len(X_train_seq)]
y_train_tabular = y_train[:len(y_train_seq)]
X_test_tabular = X_test[:seq_len]
y_test_tabular = y_test[:seq_len]

print(f"  [OK] Train: {X_train_tabular.shape}, Test: {X_test_tabular.shape}")
print(f"  [OK] Train Seq: {X_train_seq.shape}, Test Seq: {X_test_seq.shape}")

# Feature names
with open('processed_data_cicids/feature_names.txt', 'r') as f:
    feature_names = [line.strip() for line in f.readlines()]

results = []

# 1. LSTM Analizi
print("\n" + "="*80)
print("1. LSTM MODEL ANALİZİ")
print("="*80)
try:
    lstm_model = keras.models.load_model('models/best_lstm_model_cicids.keras')
    
    # Train predictions
    y_train_pred_proba = lstm_model.predict(X_train_seq, verbose=0).flatten()
    y_train_pred = (y_train_pred_proba >= 0.5).astype(int)
    
    # Test predictions
    y_test_pred_proba = lstm_model.predict(X_test_seq, verbose=0).flatten()
    y_test_pred = (y_test_pred_proba >= 0.5).astype(int)
    
    # Metrikler
    train_metrics = {
        'accuracy': accuracy_score(y_train_seq, y_train_pred) * 100,
        'precision': precision_score(y_train_seq, y_train_pred) * 100,
        'recall': recall_score(y_train_seq, y_train_pred) * 100,
        'f1': f1_score(y_train_seq, y_train_pred) * 100,
        'auc': roc_auc_score(y_train_seq, y_train_pred_proba)
    }
    
    test_metrics = {
        'accuracy': accuracy_score(y_test_seq, y_test_pred) * 100,
        'precision': precision_score(y_test_seq, y_test_pred) * 100,
        'recall': recall_score(y_test_seq, y_test_pred) * 100,
        'f1': f1_score(y_test_seq, y_test_pred) * 100,
        'auc': roc_auc_score(y_test_seq, y_test_pred_proba)
    }
    
    print(f"\n📊 LSTM - TRAIN PERFORMANSI:")
    print(f"  Accuracy:  {train_metrics['accuracy']:.2f}%")
    print(f"  Precision: {train_metrics['precision']:.2f}%")
    print(f"  Recall:    {train_metrics['recall']:.2f}%")
    print(f"  F1-Score:  {train_metrics['f1']:.2f}%")
    print(f"  AUC:       {train_metrics['auc']:.4f}")
    
    print(f"\n📊 LSTM - TEST PERFORMANSI:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.2f}%")
    print(f"  Precision: {test_metrics['precision']:.2f}%")
    print(f"  Recall:    {test_metrics['recall']:.2f}%")
    print(f"  F1-Score:  {test_metrics['f1']:.2f}%")
    print(f"  AUC:       {test_metrics['auc']:.4f}")
    
    # Overfitting kontrolü
    acc_diff = train_metrics['accuracy'] - test_metrics['accuracy']
    f1_diff = train_metrics['f1'] - test_metrics['f1']
    
    print(f"\n⚖️ OVERFITTING KONTROLÜ:")
    print(f"  Accuracy Farkı:  {acc_diff:+.2f}%")
    print(f"  F1-Score Farkı:  {f1_diff:+.2f}%")
    
    if acc_diff > 5:
        print("  ⚠️ UYARI: Belirgin overfitting tespit edildi!")
        overfitting_status = "🔴 YÜksek"
    elif acc_diff > 2:
        print("  ⚠️ Hafif overfitting mevcut")
        overfitting_status = "🟡 Orta"
    else:
        print("  ✅ Overfitting yok - Model sağlıklı!")
        overfitting_status = "🟢 Düşük"
    
    results.append({
        'Model': 'LSTM',
        'Train Acc': train_metrics['accuracy'],
        'Test Acc': test_metrics['accuracy'],
        'Acc Diff': acc_diff,
        'Train F1': train_metrics['f1'],
        'Test F1': test_metrics['f1'],
        'F1 Diff': f1_diff,
        'Overfitting': overfitting_status
    })
    
except Exception as e:
    print(f"  ❌ LSTM yuklenemedi: {e}")

# 2. CNN Analizi
print("\n" + "="*80)
print("2. CNN MODEL ANALİZİ")
print("="*80)
try:
    cnn_model = keras.models.load_model('models/best_cnn_model_cicids.keras')
    
    X_train_cnn = X_train_seq.reshape(X_train_seq.shape[0], X_train_seq.shape[1], X_train_seq.shape[2], 1)
    X_test_cnn = X_test_seq.reshape(X_test_seq.shape[0], X_test_seq.shape[1], X_test_seq.shape[2], 1)
    
    # Train predictions
    y_train_pred_proba = cnn_model.predict(X_train_cnn, verbose=0).flatten()
    y_train_pred = (y_train_pred_proba >= 0.5).astype(int)
    
    # Test predictions
    y_test_pred_proba = cnn_model.predict(X_test_cnn, verbose=0).flatten()
    y_test_pred = (y_test_pred_proba >= 0.5).astype(int)
    
    train_metrics = {
        'accuracy': accuracy_score(y_train_seq, y_train_pred) * 100,
        'f1': f1_score(y_train_seq, y_train_pred) * 100
    }
    
    test_metrics = {
        'accuracy': accuracy_score(y_test_seq, y_test_pred) * 100,
        'f1': f1_score(y_test_seq, y_test_pred) * 100
    }
    
    print(f"\n📊 CNN - TRAIN: Acc={train_metrics['accuracy']:.2f}%, F1={train_metrics['f1']:.2f}%")
    print(f"📊 CNN - TEST:  Acc={test_metrics['accuracy']:.2f}%, F1={test_metrics['f1']:.2f}%")
    
    acc_diff = train_metrics['accuracy'] - test_metrics['accuracy']
    f1_diff = train_metrics['f1'] - test_metrics['f1']
    print(f"⚖️ Accuracy Farkı: {acc_diff:+.2f}%")
    
    if acc_diff > 5:
        overfitting_status = "🔴 Yüksek"
    elif acc_diff > 2:
        overfitting_status = "🟡 Orta"
    else:
        overfitting_status = "🟢 Düşük"
    
    results.append({
        'Model': 'CNN',
        'Train Acc': train_metrics['accuracy'],
        'Test Acc': test_metrics['accuracy'],
        'Acc Diff': acc_diff,
        'Train F1': train_metrics['f1'],
        'Test F1': test_metrics['f1'],
        'F1 Diff': f1_diff,
        'Overfitting': overfitting_status
    })
    
except Exception as e:
    print(f"  ❌ CNN yuklenemedi: {e}")

# 3. LightGBM Analizi
print("\n" + "="*80)
print("3. LIGHTGBM MODEL ANALİZİ")
print("="*80)
try:
    lgbm_model = lgb.Booster(model_file='models/lightgbm_model_cicids.txt')
    
    # Train predictions
    y_train_pred_proba = lgbm_model.predict(X_train_tabular)
    y_train_pred = (y_train_pred_proba >= 0.5).astype(int)
    
    # Test predictions
    y_test_pred_proba = lgbm_model.predict(X_test_tabular)
    y_test_pred = (y_test_pred_proba >= 0.5).astype(int)
    
    train_metrics = {
        'accuracy': accuracy_score(y_train_tabular, y_train_pred) * 100,
        'f1': f1_score(y_train_tabular, y_train_pred) * 100
    }
    
    test_metrics = {
        'accuracy': accuracy_score(y_test_tabular, y_test_pred) * 100,
        'f1': f1_score(y_test_tabular, y_test_pred) * 100
    }
    
    print(f"\n📊 LightGBM - TRAIN: Acc={train_metrics['accuracy']:.2f}%, F1={train_metrics['f1']:.2f}%")
    print(f"📊 LightGBM - TEST:  Acc={test_metrics['accuracy']:.2f}%, F1={test_metrics['f1']:.2f}%")
    
    acc_diff = train_metrics['accuracy'] - test_metrics['accuracy']
    f1_diff = train_metrics['f1'] - test_metrics['f1']
    print(f"⚖️ Accuracy Farkı: {acc_diff:+.2f}%")
    
    if acc_diff > 5:
        overfitting_status = "🔴 Yüksek"
    elif acc_diff > 2:
        overfitting_status = "🟡 Orta"
    else:
        overfitting_status = "🟢 Düşük"
    
    results.append({
        'Model': 'LightGBM',
        'Train Acc': train_metrics['accuracy'],
        'Test Acc': test_metrics['accuracy'],
        'Acc Diff': acc_diff,
        'Train F1': train_metrics['f1'],
        'Test F1': test_metrics['f1'],
        'F1 Diff': f1_diff,
        'Overfitting': overfitting_status
    })
    
except Exception as e:
    print(f"  ❌ LightGBM yuklenemedi: {e}")

# 4. XGBoost Analizi
print("\n" + "="*80)
print("4. XGBOOST MODEL ANALİZİ")
print("="*80)
try:
    xgb_model = xgb.Booster()
    xgb_model.load_model('models/xgboost_model_cicids.json')
    
    # Train predictions
    dtrain = xgb.DMatrix(X_train_tabular, feature_names=feature_names)
    y_train_pred_proba = xgb_model.predict(dtrain)
    y_train_pred = (y_train_pred_proba >= 0.5).astype(int)
    
    # Test predictions
    dtest = xgb.DMatrix(X_test_tabular, feature_names=feature_names)
    y_test_pred_proba = xgb_model.predict(dtest)
    y_test_pred = (y_test_pred_proba >= 0.5).astype(int)
    
    train_metrics = {
        'accuracy': accuracy_score(y_train_tabular, y_train_pred) * 100,
        'f1': f1_score(y_train_tabular, y_train_pred) * 100
    }
    
    test_metrics = {
        'accuracy': accuracy_score(y_test_tabular, y_test_pred) * 100,
        'f1': f1_score(y_test_tabular, y_test_pred) * 100
    }
    
    print(f"\n📊 XGBoost - TRAIN: Acc={train_metrics['accuracy']:.2f}%, F1={train_metrics['f1']:.2f}%")
    print(f"📊 XGBoost - TEST:  Acc={test_metrics['accuracy']:.2f}%, F1={test_metrics['f1']:.2f}%")
    
    acc_diff = train_metrics['accuracy'] - test_metrics['accuracy']
    f1_diff = train_metrics['f1'] - test_metrics['f1']
    print(f"⚖️ Accuracy Farkı: {acc_diff:+.2f}%")
    
    if acc_diff > 5:
        overfitting_status = "🔴 Yüksek"
    elif acc_diff > 2:
        overfitting_status = "🟡 Orta"
    else:
        overfitting_status = "🟢 Düşük"
    
    results.append({
        'Model': 'XGBoost',
        'Train Acc': train_metrics['accuracy'],
        'Test Acc': test_metrics['accuracy'],
        'Acc Diff': acc_diff,
        'Train F1': train_metrics['f1'],
        'Test F1': test_metrics['f1'],
        'F1 Diff': f1_diff,
        'Overfitting': overfitting_status
    })
    
except Exception as e:
    print(f"  ❌ XGBoost yuklenemedi: {e}")

# Sonuçları DataFrame'e çevir
df = pd.DataFrame(results)

# Özet Rapor
print("\n" + "="*80)
print("📊 CICIDS2018 - OVERFİTTİNG ANALİZ RAPORU")
print("="*80)
print("\nTÜM MODELLERİN KARŞILAŞTIRMASI:")
print("-"*80)
print(df.to_string(index=False))
print("="*80)

# Görselleştirme
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('CICIDS2018 - Overfitting Analizi: Train vs Test Performansı', 
             fontsize=18, fontweight='bold', y=0.98)

# 1. Accuracy Karşılaştırması
ax1 = axes[0, 0]
x = np.arange(len(df))
width = 0.35
bars1 = ax1.bar(x - width/2, df['Train Acc'], width, label='Train Accuracy', 
                color='#3498db', alpha=0.8, edgecolor='black')
bars2 = ax1.bar(x + width/2, df['Test Acc'], width, label='Test Accuracy',
                color='#e74c3c', alpha=0.8, edgecolor='black')
ax1.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
ax1.set_title('Train vs Test Accuracy', fontweight='bold', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(df['Model'])
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([85, 105])

# Değerleri ekle
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 2. F1-Score Karşılaştırması
ax2 = axes[0, 1]
bars1 = ax2.bar(x - width/2, df['Train F1'], width, label='Train F1-Score',
                color='#2ecc71', alpha=0.8, edgecolor='black')
bars2 = ax2.bar(x + width/2, df['Test F1'], width, label='Test F1-Score',
                color='#f39c12', alpha=0.8, edgecolor='black')
ax2.set_ylabel('F1-Score (%)', fontweight='bold', fontsize=12)
ax2.set_title('Train vs Test F1-Score', fontweight='bold', fontsize=14)
ax2.set_xticks(x)
ax2.set_xticklabels(df['Model'])
ax2.legend(fontsize=11)
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim([85, 105])

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 3. Overfitting Skorları (Farklar)
ax3 = axes[1, 0]
colors_overfitting = []
for status in df['Overfitting']:
    if '🔴' in status:
        colors_overfitting.append('#e74c3c')
    elif '🟡' in status:
        colors_overfitting.append('#f39c12')
    else:
        colors_overfitting.append('#2ecc71')

bars = ax3.bar(df['Model'], df['Acc Diff'], color=colors_overfitting, 
               alpha=0.8, edgecolor='black', linewidth=2)
ax3.set_ylabel('Accuracy Difference (%)', fontweight='bold', fontsize=12)
ax3.set_title('Overfitting Score (Train - Test Accuracy)', fontweight='bold', fontsize=14)
ax3.axhline(y=2, color='orange', linestyle='--', linewidth=2, label='Orta Eşik (2%)')
ax3.axhline(y=5, color='red', linestyle='--', linewidth=2, label='Yüksek Eşik (5%)')
ax3.axhline(y=0, color='green', linestyle='-', linewidth=1, alpha=0.5)
ax3.legend(fontsize=10)
ax3.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.2,
            f'{height:+.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 4. Overfitting Durum Tablosu
ax4 = axes[1, 1]
ax4.axis('off')

table_data = []
table_data.append(['Model', 'Train Acc', 'Test Acc', 'Fark', 'Durum'])
for _, row in df.iterrows():
    table_data.append([
        row['Model'],
        f"{row['Train Acc']:.2f}%",
        f"{row['Test Acc']:.2f}%",
        f"{row['Acc Diff']:+.2f}%",
        row['Overfitting']
    ])

table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                 colWidths=[0.18, 0.20, 0.20, 0.18, 0.24])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.8)

# Header
for i in range(5):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Rows
for i in range(1, len(df) + 1):
    overfitting = table_data[i][4]
    if '🔴' in overfitting:
        row_color = '#ffe6e6'
    elif '🟡' in overfitting:
        row_color = '#fff4e6'
    else:
        row_color = '#e6ffe6'
    
    for j in range(5):
        table[(i, j)].set_facecolor(row_color)
        table[(i, j)].set_edgecolor('black')

ax4.set_title('Overfitting Durum Tablosu', fontweight='bold', fontsize=14, pad=20)

plt.tight_layout()
plt.savefig('models/cicids_overfitting_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ Grafik kaydedildi: models/cicids_overfitting_analysis.png")

# Rapor kaydet
report = f"""
{'='*80}
CICIDS2018 - OVERFİTTİNG ANALİZ RAPORU
{'='*80}

ANALIZ TARİHİ: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

1. GENEL DEĞERLENDİRME:
{'='*80}

"""

for _, row in df.iterrows():
    report += f"""
{row['Model']}:
  Train Accuracy:  {row['Train Acc']:.2f}%
  Test Accuracy:   {row['Test Acc']:.2f}%
  Accuracy Farkı:  {row['Acc Diff']:+.2f}%
  Train F1-Score:  {row['Train F1']:.2f}%
  Test F1-Score:   {row['Test F1']:.2f}%
  F1-Score Farkı:  {row['F1 Diff']:+.2f}%
  
  Overfitting Durumu: {row['Overfitting']}
{'-'*80}
"""

# Genel değerlendirme
avg_acc_diff = df['Acc Diff'].mean()
max_acc_diff = df['Acc Diff'].max()
min_acc_diff = df['Acc Diff'].min()

report += f"""

2. GENEL İSTATİSTİKLER:
{'='*80}
Ortalama Accuracy Farkı: {avg_acc_diff:.2f}%
Maksimum Accuracy Farkı: {max_acc_diff:.2f}%
Minimum Accuracy Farkı:  {min_acc_diff:.2f}%

3. SONUÇ VE ÖNERİLER:
{'='*80}
"""

if avg_acc_diff > 5:
    report += """
⚠️ YÜKSEK OVERFİTTİNG TESPİT EDİLDİ!

Öneriler:
1. Regularization artırılmalı (Dropout, L2)
2. Model karmaşıklığı azaltılmalı
3. Daha fazla training data toplanmalı
4. Data augmentation uygulanmalı
5. Cross-validation ile doğrulama yapılmalı
"""
elif avg_acc_diff > 2:
    report += """
⚠️ ORTA SEVİYEDE OVERFİTTİNG MEVCUT

Öneriler:
1. Dropout oranları gözden geçirilmeli
2. Early stopping parametreleri optimize edilmeli
3. Cross-validation ile ek doğrulama yapılmalı
"""
else:
    report += """
✅ MODELLERİN GENELLEME PERFORMANSI SAĞLIKLI!

- Train ve test performansları arasında anlamlı fark yok
- Modeller overfitting yapmadan öğrenmiş
- %100'e yakın accuracy değerleri dataset'in özelliklerinden kaynaklanıyor
- CICIDS2018 bazı saldırı tipleri için çok belirgin özellikler içeriyor

Not: Gerçek dünya uygulamalarında yeni veri ile sürekli test yapılmalı
"""

report += f"""

{'='*80}
Oluşturulan Dosyalar:
  - models/cicids_overfitting_analysis.png
  - models/cicids_overfitting_report.txt

Geliştirici: Nefise
{'='*80}
"""

with open('models/cicids_overfitting_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print("✅ Rapor kaydedildi: models/cicids_overfitting_report.txt")
print(report)

print("\n" + "="*80)
print("✅ OVERFİTTİNG ANALİZİ TAMAMLANDI!")
print("="*80)
