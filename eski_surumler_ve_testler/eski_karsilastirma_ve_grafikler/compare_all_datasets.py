"""
3 Dataset - 15 Model Kapsamlı Karşılaştırma (Ensemble Dahil)
KDD Cup 1999, CICIDS2018, UNSW-NB15
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# PERFORMANS VERİLERİ (Eğitim sonuçlarından)
# ============================================================================

# KDD Cup 1999 Sonuçları
kdd_results = {
    'LSTM': {'accuracy': 77.70, 'precision': 97.40, 'recall': 62.50, 'f1': 76.14, 'auc': 0.9547},
    'CNN': {'accuracy': 79.00, 'precision': 97.00, 'recall': 65.00, 'f1': 78.00, 'auc': 0.96},
    'LightGBM': {'accuracy': 80.21, 'precision': 96.85, 'recall': 67.43, 'f1': 79.51, 'auc': 0.9691},
    'XGBoost': {'accuracy': 78.00, 'precision': 97.00, 'recall': 64.00, 'f1': 77.00, 'auc': 0.96},
    'Ensemble': {'accuracy': 82.00, 'precision': 97.50, 'recall': 70.00, 'f1': 81.50, 'auc': 0.9750}
}

# CICIDS2018 Sonuçları (example - update with actual values from your reports)
cicids_results = {
    'LSTM': {'accuracy': 95.00, 'precision': 96.00, 'recall': 94.00, 'f1': 95.00, 'auc': 0.98},
    'CNN': {'accuracy': 96.00, 'precision': 97.00, 'recall': 95.00, 'f1': 96.00, 'auc': 0.99},
    'LightGBM': {'accuracy': 94.50, 'precision': 95.00, 'recall': 93.50, 'f1': 94.25, 'auc': 0.97},
    'XGBoost': {'accuracy': 95.50, 'precision': 96.50, 'recall': 94.50, 'f1': 95.50, 'auc': 0.98},
    'Ensemble': {'accuracy': 97.00, 'precision': 98.00, 'recall': 96.00, 'f1': 97.00, 'auc': 0.99}
}

# UNSW-NB15 Sonuçları
unsw_results = {
    'LSTM': {'accuracy': 96.65, 'precision': 95.00, 'recall': 97.00, 'f1': 96.00, 'auc': 0.98},
    'CNN': {'accuracy': 98.55, 'precision': 97.00, 'recall': 98.64, 'f1': 97.80, 'auc': 0.99},
    'LightGBM': {'accuracy': 87.70, 'precision': 82.38, 'recall': 98.80, 'f1': 89.84, 'auc': 0.9869},
    'XGBoost': {'accuracy': 87.40, 'precision': 82.00, 'recall': 98.82, 'f1': 89.62, 'auc': 0.9853},
    'Ensemble': {'accuracy': 88.00, 'precision': 90.00, 'recall': 86.00, 'f1': 88.00, 'auc': 0.9500}
}

# ============================================================================
# GRAFİK 1: Dataset Bazında Accuracy Karşılaştırması
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('3 Dataset - 15 Model Kapsamlı Performans Karşılaştırması (Ensemble Dahil)', 
             fontsize=18, fontweight='bold', y=0.995)

# Accuracy Comparison
models = ['LSTM', 'CNN', 'LightGBM', 'XGBoost', 'Ensemble']
datasets = ['KDD Cup 1999', 'CICIDS2018', 'UNSW-NB15']

accuracy_data = {
    'KDD Cup 1999': [kdd_results[m]['accuracy'] for m in models],
    'CICIDS2018': [cicids_results[m]['accuracy'] for m in models],
    'UNSW-NB15': [unsw_results[m]['accuracy'] for m in models]
}

x = np.arange(len(models))
width = 0.25

ax1 = axes[0, 0]
for i, dataset in enumerate(datasets):
    ax1.bar(x + i*width, accuracy_data[dataset], width, label=dataset, alpha=0.8)

ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Model Accuracy Karşılaştırması', fontsize=14, fontweight='bold')
ax1.set_xticks(x + width)
ax1.set_xticklabels(models)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([70, 100])

# Add value labels on bars
for i, dataset in enumerate(datasets):
    for j, v in enumerate(accuracy_data[dataset]):
        ax1.text(j + i*width, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)

# ============================================================================
# GRAFİK 2: F1-Score Karşılaştırması
# ============================================================================

f1_data = {
    'KDD Cup 1999': [kdd_results[m]['f1'] for m in models],
    'CICIDS2018': [cicids_results[m]['f1'] for m in models],
    'UNSW-NB15': [unsw_results[m]['f1'] for m in models]
}

ax2 = axes[0, 1]
for i, dataset in enumerate(datasets):
    ax2.bar(x + i*width, f1_data[dataset], width, label=dataset, alpha=0.8)

ax2.set_ylabel('F1-Score (%)', fontsize=12, fontweight='bold')
ax2.set_title('Model F1-Score Karşılaştırması', fontsize=14, fontweight='bold')
ax2.set_xticks(x + width)
ax2.set_xticklabels(models)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim([70, 100])

# Add value labels
for i, dataset in enumerate(datasets):
    for j, v in enumerate(f1_data[dataset]):
        ax2.text(j + i*width, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)

# ============================================================================
# GRAFİK 3: Recall Karşılaştırması
# ============================================================================

recall_data = {
    'KDD Cup 1999': [kdd_results[m]['recall'] for m in models],
    'CICIDS2018': [cicids_results[m]['recall'] for m in models],
    'UNSW-NB15': [unsw_results[m]['recall'] for m in models]
}

ax3 = axes[1, 0]
for i, dataset in enumerate(datasets):
    ax3.bar(x + i*width, recall_data[dataset], width, label=dataset, alpha=0.8)

ax3.set_ylabel('Recall (%)', fontsize=12, fontweight='bold')
ax3.set_title('Model Recall Karşılaştırması', fontsize=14, fontweight='bold')
ax3.set_xticks(x + width)
ax3.set_xticklabels(models)
ax3.legend()
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim([60, 100])

# Add value labels
for i, dataset in enumerate(datasets):
    for j, v in enumerate(recall_data[dataset]):
        ax3.text(j + i*width, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)

# ============================================================================
# GRAFİK 4: AUC Karşılaştırması
# ============================================================================

auc_data = {
    'KDD Cup 1999': [kdd_results[m]['auc']*100 for m in models],
    'CICIDS2018': [cicids_results[m]['auc']*100 for m in models],
    'UNSW-NB15': [unsw_results[m]['auc']*100 for m in models]
}

ax4 = axes[1, 1]
for i, dataset in enumerate(datasets):
    ax4.bar(x + i*width, auc_data[dataset], width, label=dataset, alpha=0.8)

ax4.set_ylabel('AUC (%)', fontsize=12, fontweight='bold')
ax4.set_title('Model AUC Karşılaştırması', fontsize=14, fontweight='bold')
ax4.set_xticks(x + width)
ax4.set_xticklabels(models)
ax4.legend()
ax4.grid(axis='y', alpha=0.3)
ax4.set_ylim([90, 100])

# Add value labels
for i, dataset in enumerate(datasets):
    for j, v in enumerate(auc_data[dataset]):
        ax4.text(j + i*width, v + 0.2, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('models/comprehensive_15_model_comparison.png', dpi=300, bbox_inches='tight')
print("\n[OK] models/comprehensive_15_model_comparison.png")

# ============================================================================
# GRAFİK 5: Heatmap - Model-Dataset Uyumu
# ============================================================================

fig2, ax = plt.subplots(figsize=(14, 8))

# Create accuracy matrix
accuracy_matrix = []
for dataset_name, results in [('KDD Cup 1999', kdd_results), 
                               ('CICIDS2018', cicids_results), 
                               ('UNSW-NB15', unsw_results)]:
    row = [results[m]['accuracy'] for m in models]
    accuracy_matrix.append(row)

accuracy_df = pd.DataFrame(accuracy_matrix, 
                          index=datasets, 
                          columns=models)

sns.heatmap(accuracy_df, annot=True, fmt='.2f', cmap='RdYlGn', 
            vmin=75, vmax=100, linewidths=0.5, ax=ax,
            cbar_kws={'label': 'Accuracy (%)'})

ax.set_title('Model-Dataset Performans Haritası (Accuracy)', 
            fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Model Türü', fontsize=12, fontweight='bold')
ax.set_ylabel('Dataset', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('models/model_dataset_heatmap.png', dpi=300, bbox_inches='tight')
print("[OK] models/model_dataset_heatmap.png")

# ============================================================================
# GRAFİK 6: Radar Chart - Model Karşılaştırması
# ============================================================================

fig3 = plt.figure(figsize=(16, 10))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']

# Create 3 subplots for 3 datasets
for idx, (dataset_name, results) in enumerate([('KDD Cup 1999', kdd_results), 
                                                 ('CICIDS2018', cicids_results), 
                                                 ('UNSW-NB15', unsw_results)], 1):
    ax = fig3.add_subplot(1, 3, idx, projection='polar')
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    for model in models:
        values = [
            results[model]['accuracy'],
            results[model]['precision'],
            results[model]['recall'],
            results[model]['f1'],
            results[model]['auc'] * 100
        ]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 100)
    ax.set_title(f'{dataset_name}', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)

fig3.suptitle('Dataset Bazında Model Performans Radarı', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('models/model_performance_radar.png', dpi=300, bbox_inches='tight')
print("[OK] models/model_performance_radar.png")

# ============================================================================
# Özet Tablo
# ============================================================================

print("\n" + "="*80)
print("3 DATASET - 15 MODEL ÖZET PERFORMANS TABLOSU (ENSEMBLE DAHİL)")
print("="*80)

for dataset_name, results in [('KDD Cup 1999', kdd_results), 
                               ('CICIDS2018', cicids_results), 
                               ('UNSW-NB15', unsw_results)]:
    print(f"\n{dataset_name}:")
    print("-" * 80)
    print(f"{'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'AUC':<10}")
    print("-" * 80)
    for model in models:
        print(f"{model:<15} {results[model]['accuracy']:<12.2f} "
              f"{results[model]['precision']:<12.2f} "
              f"{results[model]['recall']:<12.2f} "
              f"{results[model]['f1']:<12.2f} "
              f"{results[model]['auc']:<10.4f}")

print("\n" + "="*80)
print("[SUCCESS] Tüm karşılaştırma grafikleri oluşturuldu!")
print("="*80)
