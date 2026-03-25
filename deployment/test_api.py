"""
🧪 API Test Scripti
Flask API'sini test eder

Kullanım:
    python test_api.py
"""

import requests
import numpy as np
import json
import time
from colorama import init, Fore, Style

init(autoreset=True)

API_URL = "http://localhost:5000"

def print_header(text):
    """Başlık yazdır"""
    print("\n" + "="*80)
    print(f"{Fore.CYAN}{Style.BRIGHT}{text}")
    print("="*80)

def print_success(text):
    """Başarı mesajı"""
    print(f"{Fore.GREEN}✅ {text}")

def print_error(text):
    """Hata mesajı"""
    print(f"{Fore.RED}❌ {text}")

def print_info(text):
    """Bilgi mesajı"""
    print(f"{Fore.YELLOW}ℹ️  {text}")

def test_health():
    """Health check testi"""
    print_header("TEST 1: Health Check")
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"API aktif - Status: {data['status']}")
            print_info(f"Yüklü model sayısı: {data['models_loaded']}")
            print_info(f"Timestamp: {data['timestamp']}")
            return True
        else:
            print_error(f"HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Bağlantı hatası: {e}")
        return False

def test_models_endpoint():
    """Modeller endpoint testi"""
    print_header("TEST 2: Models Endpoint")
    
    try:
        response = requests.get(f"{API_URL}/models", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Toplam {data['total_models']} model")
            print_info(f"Önerilen model: {data['recommended']}")
            
            print("\nModel Detayları:")
            for model in data['models']:
                print(f"  • {model['name']}")
                print(f"    - Accuracy: {model['accuracy']}%")
                print(f"    - Recall: {model['recall']}%")
                print(f"    - En iyi: {model['best_for']}")
                print(f"    - Yükleme süresi: {model['load_time_ms']:.2f} ms")
            
            return True
        else:
            print_error(f"HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Hata: {e}")
        return False

def test_single_prediction():
    """Tekil tahmin testi"""
    print_header("TEST 3: Single Prediction")
    
    # Rastgele test verisi (42 özellik)
    features = np.random.randn(42).tolist()
    
    models_to_test = ['unsw_cnn', 'unsw_lightgbm', 'unsw_lstm', 'unsw_xgboost']
    
    for model_name in models_to_test:
        print(f"\n{Fore.CYAN}Testing {model_name}...")
        
        try:
            payload = {
                "features": features,
                "model": model_name
            }
            
            start_time = time.time()
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=10
            )
            request_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                
                if data['success']:
                    result = data['result']
                    print_success(f"Tahmin: {result['prediction']}")
                    print_info(f"Güven: {result['confidence']*100:.2f}%")
                    print_info(f"Model tahmin süresi: {result['prediction_time_ms']:.2f} ms")
                    print_info(f"Toplam istek süresi: {request_time:.2f} ms")
                else:
                    print_error("Tahmin başarısız")
            else:
                print_error(f"HTTP {response.status_code}")
                
        except Exception as e:
            print_error(f"Hata: {e}")
    
    return True

def test_batch_prediction():
    """Toplu tahmin testi"""
    print_header("TEST 4: Batch Prediction")
    
    # 100 örnek
    num_samples = 100
    features = np.random.randn(num_samples, 42).tolist()
    
    print_info(f"{num_samples} örnek ile test ediliyor...")
    
    try:
        payload = {
            "features": features,
            "model": "unsw_lightgbm"  # En hızlı model
        }
        
        start_time = time.time()
        response = requests.post(
            f"{API_URL}/predict/batch",
            json=payload,
            timeout=60
        )
        total_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            
            if data['success']:
                print_success(f"Toplam örnek: {data['total_samples']}")
                print_success(f"Başarılı tahmin: {data['successful_predictions']}")
                print_info(f"Saldırı tespit: {data['attacks_detected']}")
                print_info(f"Normal tespit: {data['normal_detected']}")
                print_info(f"Toplam süre: {total_time:.2f} ms")
                print_info(f"Örnek başına: {total_time/num_samples:.2f} ms")
            else:
                print_error("Batch tahmin başarısız")
        else:
            print_error(f"HTTP {response.status_code}")
            
    except Exception as e:
        print_error(f"Hata: {e}")
    
    return True

def test_stats_endpoint():
    """İstatistik endpoint testi"""
    print_header("TEST 5: Stats Endpoint")
    
    try:
        response = requests.get(f"{API_URL}/stats", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print_success("İstatistikler alındı")
            print_info(f"Toplam tahmin: {data['total_predictions']}")
            print_info(f"Ortalama tahmin süresi: {data['average_prediction_time_ms']:.2f} ms")
            print_info(f"Timestamp: {data['timestamp']}")
            return True
        else:
            print_error(f"HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Hata: {e}")
        return False

def test_performance():
    """Performans testi"""
    print_header("TEST 6: Performance Test")
    
    print_info("100 ardışık tahmin yapılıyor...")
    
    features = np.random.randn(42).tolist()
    times = []
    
    for i in range(100):
        try:
            payload = {
                "features": features,
                "model": "unsw_lightgbm"
            }
            
            start_time = time.time()
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=10
            )
            request_time = (time.time() - start_time) * 1000
            times.append(request_time)
            
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/100 tamamlandı...")
                
        except Exception as e:
            print_error(f"Hata: {e}")
    
    if times:
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        print_success(f"100 tahmin tamamlandı")
        print_info(f"Ortalama süre: {avg_time:.2f} ms")
        print_info(f"En hızlı: {min_time:.2f} ms")
        print_info(f"En yavaş: {max_time:.2f} ms")
        print_info(f"Saniyede tahmin: {1000/avg_time:.2f}")
    
    return True

def test_error_handling():
    """Hata yönetimi testi"""
    print_header("TEST 7: Error Handling")
    
    # Test 1: Eksik feature
    print(f"\n{Fore.CYAN}Test 7.1: Eksik feature")
    try:
        payload = {"features": [1, 2, 3]}  # Sadece 3 feature
        response = requests.post(f"{API_URL}/predict", json=payload)
        
        if response.status_code == 400:
            print_success("Eksik feature hatası doğru yakalandı")
        else:
            print_error(f"Beklenmeyen durum kodu: {response.status_code}")
    except Exception as e:
        print_error(f"Hata: {e}")
    
    # Test 2: Geçersiz model
    print(f"\n{Fore.CYAN}Test 7.2: Geçersiz model")
    try:
        payload = {
            "features": np.random.randn(42).tolist(),
            "model": "nonexistent_model"
        }
        response = requests.post(f"{API_URL}/predict", json=payload)
        
        if response.status_code == 500:
            print_success("Geçersiz model hatası doğru yakalandı")
        else:
            print_error(f"Beklenmeyen durum kodu: {response.status_code}")
    except Exception as e:
        print_error(f"Hata: {e}")
    
    # Test 3: Eksik payload
    print(f"\n{Fore.CYAN}Test 7.3: Eksik payload")
    try:
        response = requests.post(f"{API_URL}/predict", json={})
        
        if response.status_code == 400:
            print_success("Eksik payload hatası doğru yakalandı")
        else:
            print_error(f"Beklenmeyen durum kodu: {response.status_code}")
    except Exception as e:
        print_error(f"Hata: {e}")

def run_all_tests():
    """Tüm testleri çalıştır"""
    print(f"\n{Fore.MAGENTA}{Style.BRIGHT}")
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║                     🧪 API TEST SÜİTİ BAŞLATILIYOR                          ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    
    print(f"\n{Fore.YELLOW}API URL: {API_URL}")
    print(f"{Fore.YELLOW}Test Zamanı: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    results = {
        "Health Check": test_health(),
        "Models Endpoint": test_models_endpoint(),
        "Single Prediction": test_single_prediction(),
        "Batch Prediction": test_batch_prediction(),
        "Stats Endpoint": test_stats_endpoint(),
        "Performance": test_performance(),
        "Error Handling": test_error_handling()
    }
    
    # Özet
    print_header("📊 TEST SONUÇLARI")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = f"{Fore.GREEN}✅ PASSED" if result else f"{Fore.RED}❌ FAILED"
        print(f"{test_name:.<60} {status}")
    
    print("\n" + "="*80)
    print(f"{Fore.CYAN}Toplam: {passed}/{total} test başarılı")
    
    if passed == total:
        print(f"{Fore.GREEN}{Style.BRIGHT}🎉 TÜM TESTLER BAŞARILI!")
    else:
        print(f"{Fore.YELLOW}⚠️  Bazı testler başarısız oldu")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    run_all_tests()
