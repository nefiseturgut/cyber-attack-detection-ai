"""
LightGBM Siber Saldırı Tespit Modeli
Gradient Boosting algoritması ile hızlı ve etkili saldırı tespiti
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime


class CyberAttackLightGBM:
    """LightGBM tabanlı siber saldırı tespit modeli"""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.feature_importance = None
        
    def load_data(self, data_dir='processed_data'):
        """
        İşlenmiş veriyi yükle
        """
        print("\n" + "="*80)
        print("📂 VERİ YÜKLEME")
        print("="*80)
        
        # Veriyi yükle
        X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
        y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
        X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
        
        # Feature isimleri
        with open(os.path.join(data_dir, 'feature_names.txt'), 'r') as f:
            self.feature_names = [line.strip() for line in f.readlines()]
        
        print(f"\n✅ Veri yüklendi:")
        print(f"   Train: {X_train.shape}")
        print(f"   Test: {X_test.shape}")
        print(f"   Features: {len(self.feature_names)}")
        
        # Sınıf dağılımı
        print(f"\n📊 Sınıf Dağılımı:")
        print(f"   Train - Normal: {(y_train==0).sum():,}, Saldırı: {(y_train==1).sum():,}")
        print(f"   Test - Normal: {(y_test==0).sum():,}, Saldırı: {(y_test==1).sum():,}")
        
        return X_train, y_train, X_test, y_test
    
    def train(self, X_train, y_train, X_val=None, y_val=None, params=None):
        """
        LightGBM modelini eğit
        
        Args:
            X_train: Eğitim özellikleri
            y_train: Eğitim etiketleri
            X_val: Validation özellikleri (opsiyonel)
            y_val: Validation etiketleri (opsiyonel)
            params: Model parametreleri (opsiyonel)
        """
        print("\n" + "="*80)
        print("🎓 LIGHTGBM MODEL EĞİTİMİ")
        print("="*80)
        
        # Default parametreler
        if params is None:
            params = {
                'objective': 'binary',
                'metric': ['binary_logloss', 'auc', 'binary_error'],
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': 0,
                'max_depth': -1,
                'min_data_in_leaf': 20,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1
            }
        
        print(f"\n⚙️  Model Parametreleri:")
        for key, value in params.items():
            print(f"   {key}: {value}")
        
        # Dataset oluştur
        train_data = lgb.Dataset(
            X_train, 
            label=y_train,
            feature_name=self.feature_names
        )
        
        # Validation set varsa
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(
                X_val,
                label=y_val,
                feature_name=self.feature_names,
                reference=train_data
            )
            valid_sets.append(val_data)
            valid_names.append('valid')
            print(f"\n📊 Validation seti kullanılıyor: {X_val.shape}")
        
        # Class weights hesapla
        n_normal = (y_train == 0).sum()
        n_attack = (y_train == 1).sum()
        scale_pos_weight = n_normal / n_attack
        
        params['scale_pos_weight'] = scale_pos_weight
        print(f"\n⚖️  Scale pos weight: {scale_pos_weight:.2f}")
        
        # Eğitim
        print(f"\n🏃 Eğitim başlıyor...")
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )
        
        print(f"\n✅ Eğitim tamamlandı!")
        print(f"   Best iteration: {self.model.best_iteration}")
        print(f"   Best score: {self.model.best_score}")
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔝 En Önemli 10 Özellik:")
        for idx, row in self.feature_importance.head(10).iterrows():
            print(f"   {row['feature']:30} : {row['importance']:10.2f}")
        
        return self
    
    def evaluate(self, X_test, y_test):
        """
        Modeli değerlendir
        
        Args:
            X_test: Test özellikleri
            y_test: Test etiketleri
        """
        print("\n" + "="*80)
        print("📊 MODEL DEĞERLENDİRME")
        print("="*80)
        
        # Tahmin yap
        print("\n🔮 Tahminler yapılıyor...")
        y_pred_proba = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Metrikler
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n📈 Test Sonuçları:")
        print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"   F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
        print(f"   AUC:       {auc:.4f}")
        
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
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\n✅ Doğru Tahminler:")
        print(f"   True Negatives (Normal→Normal):   {tn:,}")
        print(f"   True Positives (Saldırı→Saldırı): {tp:,}")
        
        print(f"\n❌ Yanlış Tahminler:")
        print(f"   False Positives (Normal→Saldırı): {fp:,}")
        print(f"   False Negatives (Saldırı→Normal): {fn:,}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def plot_feature_importance(self, top_n=20, save_path='models/lgbm_feature_importance.png'):
        """Feature importance grafiği"""
        if self.feature_importance is None:
            print("⚠️  Model henüz eğitilmedi!")
            return
        
        plt.figure(figsize=(12, 8))
        
        top_features = self.feature_importance.head(top_n)
        plt.barh(range(len(top_features)), top_features['importance'].values)
        plt.yticks(range(len(top_features)), top_features['feature'].values)
        plt.xlabel('Importance (Gain)', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title(f'Top {top_n} Feature Importance - LightGBM', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n📊 Feature importance grafiği kaydedildi: {save_path}")
    
    def plot_confusion_matrix(self, cm, save_path='models/lgbm_confusion_matrix.png'):
        """Confusion matrix görselleştir"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Normal', 'Saldırı'],
            yticklabels=['Normal', 'Saldırı'],
            cbar_kws={'label': 'Sayı'}
        )
        plt.title('LightGBM - Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('Gerçek Etiket', fontsize=12)
        plt.xlabel('Tahmin Edilen Etiket', fontsize=12)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Confusion matrix kaydedildi: {save_path}")
    
    def plot_roc_curve(self, y_test, y_pred_proba, save_path='models/lgbm_roc_curve.png'):
        """ROC curve çiz"""
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'LightGBM (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - LightGBM', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 ROC curve kaydedildi: {save_path}")
    
    def save_model(self, path='models/lightgbm_model.txt'):
        """Modeli kaydet"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)
        print(f"\n💾 Model kaydedildi: {path}")
        
        # Feature importance da kaydet
        importance_path = path.replace('.txt', '_feature_importance.csv')
        self.feature_importance.to_csv(importance_path, index=False)
        print(f"💾 Feature importance kaydedildi: {importance_path}")
    
    def load_model(self, path='models/lightgbm_model.txt'):
        """Modeli yükle"""
        self.model = lgb.Booster(model_file=path)
        print(f"✅ Model yüklendi: {path}")
        return self


def main():
    """Ana fonksiyon - LightGBM modeli eğitimi"""
    print("\n" + "="*80)
    print("🚀 LIGHTGBM SİBER SALDIRI TESPİT MODELİ")
    print("="*80 + "\n")
    
    # Model oluştur
    lgbm_model = CyberAttackLightGBM()
    
    # Veriyi yükle
    X_train, y_train, X_test, y_test = lgbm_model.load_data()
    
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
    
    # Model eğit
    lgbm_model.train(X_train, y_train, X_val, y_val)
    
    # Değerlendir
    results = lgbm_model.evaluate(X_test, y_test)
    
    # Grafikleri kaydet
    lgbm_model.plot_feature_importance(top_n=20)
    lgbm_model.plot_confusion_matrix(results['confusion_matrix'])
    lgbm_model.plot_roc_curve(y_test, results['y_pred_proba'])
    
    # Modeli kaydet
    lgbm_model.save_model()
    
    print("\n" + "="*80)
    print("✨ LIGHTGBM MODELİ EĞİTİMİ TAMAMLANDI!")
    print("="*80)
    print(f"\n🎯 Final Test Metrics:")
    print(f"   Accuracy:  {results['accuracy']*100:.2f}%")
    print(f"   Precision: {results['precision']*100:.2f}%")
    print(f"   Recall:    {results['recall']*100:.2f}%")
    print(f"   F1-Score:  {results['f1_score']*100:.2f}%")
    print(f"   AUC:       {results['auc']:.4f}")
    
    print(f"\n📁 Model ve grafikler 'models/' klasöründe kaydedildi")
    print(f"\n⚡ LightGBM modeli hızlı ve etkili saldırı tespiti yapabilir!")
    
    return lgbm_model, results


if __name__ == "__main__":
    lgbm_model, results = main()
