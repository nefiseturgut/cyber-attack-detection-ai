"""
Gerçek Klasör Yapısını Analiz Et ve Raporla
"""

import os
from pathlib import Path

def get_directory_structure(path, indent="", max_depth=3, current_depth=0):
    """Klasör yapısını analiz et"""
    structure = []
    
    if current_depth >= max_depth:
        return structure
    
    try:
        items = sorted(os.listdir(path))
        dirs = [item for item in items if os.path.isdir(os.path.join(path, item)) and not item.startswith('.')]
        files = [item for item in items if os.path.isfile(os.path.join(path, item))]
        
        # Klasörler
        for directory in dirs:
            dir_path = os.path.join(path, directory)
            num_items = len([x for x in os.listdir(dir_path) if not x.startswith('.')])
            structure.append(f"{indent}├── 📁 {directory}/ ({num_items} items)")
            structure.extend(get_directory_structure(dir_path, indent + "│   ", max_depth, current_depth + 1))
        
        # Dosyalar
        for file in files:
            file_path = os.path.join(path, file)
            size = os.path.getsize(file_path)
            size_str = f"{size/1024:.1f} KB" if size > 1024 else f"{size} bytes"
            
            # Dosya tipine göre emoji
            if file.endswith('.py'):
                emoji = '🐍'
            elif file.endswith('.md'):
                emoji = '📄'
            elif file.endswith('.txt'):
                emoji = '📝'
            elif file.endswith('.pdf'):
                emoji = '📕'
            elif file.endswith('.png'):
                emoji = '🖼️'
            elif file.endswith(('.npy', '.npz')):
                emoji = '💾'
            elif file.endswith(('.keras', '.h5')):
                emoji = '🤖'
            elif file.endswith('.json'):
                emoji = '📋'
            elif file.endswith('.csv'):
                emoji = '📊'
            else:
                emoji = '📄'
            
            structure.append(f"{indent}├── {emoji} {file} ({size_str})")
    
    except PermissionError:
        structure.append(f"{indent}├── ⚠️ [Permission Denied]")
    
    return structure

# Ana klasörü analiz et
project_root = r"c:\Users\Nefise\siber_saldırı_project"
print("="*80)
print("🛡️ SİBER SALDIRI TESPİT PROJESİ - GERÇEK KLASÖR YAPISI")
print("="*80)
print(f"\n📂 Root: {project_root}\n")

structure = get_directory_structure(project_root)
for line in structure:
    print(line)

# Özet istatistikler
print("\n" + "="*80)
print("📊 PROJE İSTATİSTİKLERİ")
print("="*80)

# Dosya sayıları
file_counts = {
    'Python Scripts': 0,
    'Models': 0,
    'Images': 0,
    'Data Files': 0,
    'Documents': 0,
    'Total Files': 0
}

for root, dirs, files in os.walk(project_root):
    # .venv klasörünü atla
    if '.venv' in root or '__pycache__' in root:
        continue
    
    for file in files:
        file_counts['Total Files'] += 1
        if file.endswith('.py'):
            file_counts['Python Scripts'] += 1
        elif file.endswith(('.keras', '.h5', '.txt', '.json')) and 'model' in file.lower():
            file_counts['Models'] += 1
        elif file.endswith(('.png', '.jpg', '.jpeg')):
            file_counts['Images'] += 1
        elif file.endswith(('.npy', '.npz', '.csv')):
            file_counts['Data Files'] += 1
        elif file.endswith(('.md', '.pdf', '.txt')) and 'model' not in file.lower():
            file_counts['Documents'] += 1

for category, count in file_counts.items():
    print(f"{category:20} : {count:3}")

print("\n" + "="*80)

# Dosyaları türe göre listele
print("\n🐍 PYTHON SCRIPTLERI:")
for root, dirs, files in os.walk(project_root):
    if '.venv' in root or '__pycache__' in root:
        continue
    for file in sorted(files):
        if file.endswith('.py'):
            print(f"   ✓ {file}")

print("\n🤖 EĞİTİLMİŞ MODELLER:")
models_dir = os.path.join(project_root, 'models')
if os.path.exists(models_dir):
    for file in sorted(os.listdir(models_dir)):
        if file.endswith(('.keras', '.h5', '.json', '.txt')) and 'model' in file.lower():
            size = os.path.getsize(os.path.join(models_dir, file))
            print(f"   ✓ {file} ({size/1024:.1f} KB)")

print("\n📊 GÖRSELLEŞTİRMELER:")
if os.path.exists(models_dir):
    for file in sorted(os.listdir(models_dir)):
        if file.endswith('.png'):
            size = os.path.getsize(os.path.join(models_dir, file))
            print(f"   ✓ {file} ({size/1024:.1f} KB)")

print("\n📄 DOKÜMANTASYON:")
for file in sorted(os.listdir(project_root)):
    if file.endswith(('.md', '.pdf', '.txt')) and os.path.isfile(os.path.join(project_root, file)):
        size = os.path.getsize(os.path.join(project_root, file))
        print(f"   ✓ {file} ({size/1024 if size > 1024 else size} {'KB' if size > 1024 else 'bytes'})")

print("\n✅ Klasör yapısı analizi tamamlandı!")
