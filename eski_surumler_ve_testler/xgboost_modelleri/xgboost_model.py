"""
XGBoost Siber Saldırı Tespit Modeli
Gradient Boosting ile yüksek performanslı saldırı tespiti
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime


class CyberAttackXGBoost:
    """XGBoost tabanlı siber saldırı tespit modeli"""
    
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
        XGBoost modelini eğit
        
        Args:
            X_train: Eğitim özellikleri
            y_train: Eğitim etiketleri
            X_val: Validation özellikleri (opsiyonel)
            y_val: Validation etiketleri (opsiyonel)
            params: Model parametreleri (opsiyonel)
        """
        print("\n" + "="*80)
        print("🎓 XGBOOST MODEL EĞİTİMİ")
        print("="*80)
        
        # Default parametreler
        if params is None:
            params = {
                'objective': 'binary:logistic',
                'eval_metric': ['logloss', 'auc', 'error'],
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 500,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'tree_method': 'hist',
                'verbosity': 1
            }
        
        print(f"\n⚙️  Model Parametreleri:")
        for key, value in params.items():
            if key not in ['eval_metric']:
                print(f"   {key}: {value}")
        
        # Class weights hesapla
        n_normal = (y_train == 0).sum()
        n_attack = (y_train == 1).sum()
        scale_pos_weight = n_normal / n_attack
        
        params['scale_pos_weight'] = scale_pos_weight
        print(f"\n⚖️  Scale pos weight: {scale_pos_weight:.2f}")
        
        # DMatrix oluştur
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
        
        # Validation set
        evals = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
            evals.append((dval, 'valid'))
            print(f"\n📊 Validation seti kullanılıyor: {X_val.shape}")
        
        # Eğitim
        print(f"\n🏃 Eğitim başlıyor...")
        
        # Extract n_estimators for num_boost_round
        num_boost_round = params.pop('n_estimators', 500)
        
        evals_result = {}
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=100,
            evals_result=evals_result
        )
        
        print(f"\n✅ Eğitim tamamlandı!")
        print(f"   Best iteration: {self.model.best_iteration}")
        print(f"   Best score: {self.model.best_score}")
        
        # Feature importance
        importance_dict = self.model.get_score(importance_type='gain')
        self.feature_importance = pd.DataFrame([
            {'feature': k, 'importance': v} 
            for k, v in importance_dict.items()
        ]).sort_values('importance', ascending=False)
        
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
        
        # DMatrix oluştur
        dtest = xgb.DMatrix(X_test, feature_names=self.feature_names)
        
        # Tahmin yap
        print("\n🔮 Tahminler yapılıyor...")
        y_pred_proba = self.model.predict(dtest, iteration_range=(0, self.model.best_iteration + 1))
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
    
    def plot_feature_importance(self, top_n=20, save_path='models/xgb_feature_importance.png'):
        """Feature importance grafiği"""
        if self.feature_importance is None:
            print("⚠️  Model henüz eğitilmedi!")
            return
        
        plt.figure(figsize=(12, 8))
        
        top_features = self.feature_importance.head(top_n)
        plt.barh(range(len(top_features)), top_features['importance'].values, color='#e74c3c')
        plt.yticks(range(len(top_features)), top_features['feature'].values)
        plt.xlabel('Importance (Gain)', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title(f'Top {top_n} Feature Importance - XGBoost', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n📊 Feature importance grafiği kaydedildi: {save_path}")
    
    def plot_confusion_matrix(self, cm, save_path='models/xgb_confusion_matrix.png'):
        """Confusion matrix görselleştir"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Normal', 'Saldırı'],
            yticklabels=['Normal', 'Saldırı'],
            cbar_kws={'label': 'Sayı'}
        )
        plt.title('XGBoost - Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('Gerçek Etiket', fontsize=12)
        plt.xlabel('Tahmin Edilen Etiket', fontsize=12)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Confusion matrix kaydedildi: {save_path}")
    
    def plot_roc_curve(self, y_test, y_pred_proba, save_path='models/xgb_roc_curve.png'):
        """ROC curve çiz"""
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'XGBoost (AUC = {auc_score:.4f})', color='#e74c3c')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - XGBoost', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 ROC curve kaydedildi: {save_path}")
    
    def plot_training_history(self, save_path='models/xgb_training_history.png'):
        """Eğitim geçmişini görselleştir"""
        if not hasattr(self.model, 'evals_result'):
            print("⚠️  Eğitim geçmişi bulunamadı!")
            return
        
        evals_result = self.model.evals_result()
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle('XGBoost Training History', fontsize=16, fontweight='bold')
        
        # AUC
        if 'train' in evals_result and 'auc' in evals_result['train']:
            axes[0].plot(evals_result['train']['auc'], label='Train', linewidth=2)
            if 'valid' in evals_result:
                axes[0].plot(evals_result['valid']['auc'], label='Validation', linewidth=2)
            axes[0].set_title('AUC Score', fontweight='bold')
            axes[0].set_xlabel('Iteration')
            axes[0].set_ylabel('AUC')
            axes[0].legend()
            axes[0].grid(alpha=0.3)
        
        # Log Loss
        if 'train' in evals_result and 'logloss' in evals_result['train']:
            axes[1].plot(evals_result['train']['logloss'], label='Train', linewidth=2)
            if 'valid' in evals_result:
                axes[1].plot(evals_result['valid']['logloss'], label='Validation', linewidth=2)
            axes[1].set_title('Log Loss', fontweight='bold')
            axes[1].set_xlabel('Iteration')
            axes[1].set_ylabel('Log Loss')
            axes[1].legend()
            axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training history grafiği kaydedildi: {save_path}")
    
    def save_model(self, path='models/xgboost_model.json'):
        """Modeli kaydet"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)
        print(f"\n💾 Model kaydedildi: {path}")
        
        # Feature importance da kaydet
        importance_path = path.replace('.json', '_feature_importance.csv')
        self.feature_importance.to_csv(importance_path, index=False)
        print(f"💾 Feature importance kaydedildi: {importance_path}")
    
    def load_model(self, path='models/xgboost_model.json'):
        """Modeli yükle"""
        self.model = xgb.Booster()
        self.model.load_model(path)
        print(f"✅ Model yüklendi: {path}")
        return self


def main():
    """Ana fonksiyon - XGBoost modeli eğitimi"""
    print("\n" + "="*80)
    print("🚀 XGBOOST SİBER SALDIRI TESPİT MODELİ")
    print("="*80 + "\n")
    
    # Model oluştur
    xgb_model = CyberAttackXGBoost()
    
    # Veriyi yükle
    X_train, y_train, X_test, y_test = xgb_model.load_data()
    
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
    xgb_model.train(X_train, y_train, X_val, y_val)
    
    # Değerlendir
    results = xgb_model.evaluate(X_test, y_test)
    
    # Grafikleri kaydet
    xgb_model.plot_feature_importance(top_n=20)
    xgb_model.plot_confusion_matrix(results['confusion_matrix'])
    xgb_model.plot_roc_curve(y_test, results['y_pred_proba'])
    xgb_model.plot_training_history()
    
    # Modeli kaydet
    xgb_model.save_model()
    
    print("\n" + "="*80)
    print("✨ XGBOOST MODELİ EĞİTİMİ TAMAMLANDI!")
    print("="*80)
    print(f"\n🎯 Final Test Metrics:")
    print(f"   Accuracy:  {results['accuracy']*100:.2f}%")
    print(f"   Precision: {results['precision']*100:.2f}%")
    print(f"   Recall:    {results['recall']*100:.2f}%")
    print(f"   F1-Score:  {results['f1_score']*100:.2f}%")
    print(f"   AUC:       {results['auc']:.4f}")
    
    print(f"\n📁 Model ve grafikler 'models/' klasöründe kaydedildi")
    print(f"\n⚡ XGBoost modeli hızlı ve yüksek performanslı saldırı tespiti yapabilir!")
    
    return xgb_model, results


if __name__ == "__main__":
    xgb_model, results = main()
