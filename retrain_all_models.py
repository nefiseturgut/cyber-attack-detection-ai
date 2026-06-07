import subprocess
import time
import os

models_to_run = [
    # Önce en hzl olanlar (LightGBM) eitelim (Maks 3-5 dakika)
    "lightgbm_model.py",
    "lightgbm_model_cicids.py",
    "lightgbm_model_unsw.py",
    
    # Sonra CNN Modellerini eitelim
    "cnn_model.py",
    "cnn_model_cicids.py",
    "cnn_model_unsw.py",
    
    # En uzun süren LSTM modellerini sona saklayalm
    "lstm_model.py",
    "lstm_model_cicids.py",
    "lstm_model_unsw.py"
]

def run_models():
    print("="*60)
    print(" TÜM MODELLER ÇN YENDEN ETM BALATICI ".center(60))
    print("="*60 + "\n")
    
    total_start_time = time.time()
    
    for idx, model_script in enumerate(models_to_run, 1):
        if not os.path.exists(model_script):
            print(f"[{idx}/{len(models_to_run)}]  UYARI: {model_script} dosyas bulunamad, atlanyor...")
            continue
            
        print(f"\n[{idx}/{len(models_to_run)}]  ÇALITIRILIYOR: {model_script} ...")
        print("-" * 60)
        
        start_time = time.time()
        
        try:
            # Subprocess ile senkron olarak çaltr, çktlar direkt konsolda göster
            process = subprocess.Popen(
                ["python", model_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Gerçek zamanl konsol çkts
            for line in process.stdout:
                print(line, end="")
                
            process.wait()
            
            # Çökme kontrolü
            if process.returncode != 0:
                print(f"\n HATA: {model_script} çalrken bir hatayla durdu!")
                continue
                
        except Exception as e:
            print(f"\n HATA PENCERES: {model_script} balatlamad: {str(e)}")
            continue
            
        elapsed = time.time() - start_time
        print("-" * 60)
        print(f" BTT: {model_script} (Süre: {elapsed/60:.2f} dakika)\n")
        
    total_elapsed = time.time() - total_start_time
    print("="*60)
    print(" TÜM ETMLER BAARIYLA TAMAMLANDI! ".center(60))
    print(f"Toplam Geçen Süre: {total_elapsed/60:.2f} dakika".center(60))
    print("="*60)

if __name__ == "__main__":
    run_models()
