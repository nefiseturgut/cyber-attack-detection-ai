import subprocess
import time
import os

models_to_run = [
    # Önce en hızlı olanları (LightGBM) eğitelim (Maks 3-5 dakika)
    "lightgbm_model.py",
    "lightgbm_model_cicids.py",
    "lightgbm_model_unsw.py",
    
    # Sonra CNN Modellerini eğitelim
    "cnn_model.py",
    "cnn_model_cicids.py",
    "cnn_model_unsw.py",
    
    # En uzun süren LSTM modellerini sona saklayalım
    "lstm_model.py",
    "lstm_model_cicids.py",
    "lstm_model_unsw.py"
]

def run_models():
    print("="*60)
    print("🚀 TÜM MODELLER İÇİN YENİDEN EĞİTİM BAŞLATICI 🚀".center(60))
    print("="*60 + "\n")
    
    total_start_time = time.time()
    
    for idx, model_script in enumerate(models_to_run, 1):
        if not os.path.exists(model_script):
            print(f"[{idx}/{len(models_to_run)}] ⚠️ UYARI: {model_script} dosyası bulunamadı, atlanıyor...")
            continue
            
        print(f"\n[{idx}/{len(models_to_run)}] 🔄 ÇALIŞTIRILIYOR: {model_script} ...")
        print("-" * 60)
        
        start_time = time.time()
        
        try:
            # Subprocess ile senkron olarak çalıştır, çıktıları direkt konsolda göster
            process = subprocess.Popen(
                ["python", model_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Gerçek zamanlı konsol çıktısı
            for line in process.stdout:
                print(line, end="")
                
            process.wait()
            
            # Çökme kontrolü
            if process.returncode != 0:
                print(f"\n❌ HATA: {model_script} çalışırken bir hatayla durdu!")
                continue
                
        except Exception as e:
            print(f"\n❌ HATA PENCERESİ: {model_script} başlatılamadı: {str(e)}")
            continue
            
        elapsed = time.time() - start_time
        print("-" * 60)
        print(f"✅ BİTTİ: {model_script} (Süre: {elapsed/60:.2f} dakika)\n")
        
    total_elapsed = time.time() - total_start_time
    print("="*60)
    print("🎊 TÜM EĞİTİMLER BAŞARIYLA TAMAMLANDI! 🎊".center(60))
    print(f"Toplam Geçen Süre: {total_elapsed/60:.2f} dakika".center(60))
    print("="*60)

if __name__ == "__main__":
    run_models()
