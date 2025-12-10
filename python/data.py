import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# NSL-KDD verisetini yükleme
def load_nsl_kdd():
    """NSL-KDD verisetini yükler"""
    
    # CSV dosyalarını yükle
    try:
        # Train ve test dosyalarını yükle
        train_df = pd.read_csv('KDDTrain+.txt', header=None)
        test_df = pd.read_csv('KDDTest+.txt', header=None)
        
        print("✅ Verisetleri başarıyla yüklendi!")
        return train_df, test_df
        
    except FileNotFoundError:
        print("❌ Dosyalar bulunamadı! Lütfen indirme linklerini kullanın.")
        return None, None

# Özellik isimlerini tanımla
feature_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'difficulty_level'
]

# Veriyi yükle
train_df, test_df = load_nsl_kdd()

if train_df is not None:
    # Kolon isimlerini ata
    train_df.columns = feature_names
    test_df.columns = feature_names
    
    print("📊 Train verisi şekli:", train_df.shape)
    print("📊 Test verisi şekli:", test_df.shape)