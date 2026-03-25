"""
Tüm Datasetler için Preprocessing Master Script
Bu script tüm 3 dataseti sırayla işler
"""

import subprocess
import sys
from pathlib import Path


def run_script(script_name):
    """Bir scripti çalıştır"""
    print(f"\n{'='*80}")
    print(f"🚀 {script_name} çalıştırılıyor...")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=600  # 10 dakika timeout
        )
        
        print(result.stdout)
        
        if result.returncode == 0:
            print(f"✅ {script_name} başarıyla tamamlandı!")
            return True
        else:
            print(f"❌ {script_name} hata verdi!")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {script_name} timeout oldu (10 dakika)")
        return False
    except Exception as e:
        print(f"❌ Hata: {e}")
        return False


def main():
    """Tüm preprocessing scriptlerini sırayla çalıştır"""
    print("\n" + "="*80)
    print("🎯 TÜM DATASETLER İÇİN PREPROCESSİNG")
    print("="*80)
    
    scripts = [
        ('data_preprocessing.py', 'KDD Cup 1999'),
        ('data_preprocessing_cicids.py', 'CICIDS2018'),
        ('data_preprocessing_unsw.py', 'UNSW_NB15'),
    ]
    
    results = {}
    
    for script, dataset_name in scripts:
        if Path(script).exists():
            print(f"\n📊 Dataset: {dataset_name}")
            success = run_script(script)
            results[dataset_name] = success
        else:
            print(f"⚠️ {script} bulunamadı, atlanıyor...")
            results[dataset_name] = False
    
    # Özet
    print("\n" + "="*80)
    print("📊 PREPROCESSİNG ÖZET")
    print("="*80)
    
    for dataset, success in results.items():
        status = "✅ BAŞARILI" if success else "❌ BAŞARISIZ"
        print(f"{dataset:20} : {status}")
    
    # Başarı kontrolü
    successful = sum(results.values())
    total = len(results)
    
    print(f"\nToplam: {successful}/{total} dataset başarıyla işlendi")
    
    if successful == total:
        print("\n🎉 TÜM DATASETLER HAZIR!")
        print("   Şimdi model eğitimi için hazırsınız!")
    else:
        print("\n⚠️ Bazı datasetlerde sorun var, kontrol edin.")


if __name__ == "__main__":
    main()
