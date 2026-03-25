"""
Markdown'ı HTML'e çevirip tarayıcıda yazdır
"""

import markdown
import os
import webbrowser

# Markdown dosyasını oku
print("📖 Markdown dosyası okunuyor...")
with open('FINAL_PROJECT_REPORT.md', 'r', encoding='utf-8') as f:
    md_content = f.read()

# Markdown'ı HTML'e çevir
print("🔄 HTML'e çevriliyor...")
html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code', 'nl2br'])

# Güzel görünümlü HTML oluştur
html_with_style = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Siber Saldırı Tespit Sistemi - Final Rapor</title>
    <style>
        @media print {{
            @page {{
                size: A4;
                margin: 1.5cm;
            }}
        }}
        
        body {{
            font-family: 'Segoe UI', 'Arial', sans-serif;
            line-height: 1.8;
            color: #2c3e50;
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 20px;
            background: #fff;
        }}
        
        h1 {{
            color: #1f77b4;
            border-bottom: 4px solid #1f77b4;
            padding-bottom: 15px;
            margin-top: 40px;
            margin-bottom: 25px;
            font-size: 2.5em;
            page-break-after: avoid;
        }}
        
        h2 {{
            color: #2980b9;
            border-bottom: 2px solid #bdc3c7;
            padding-bottom: 10px;
            margin-top: 35px;
            margin-bottom: 20px;
            font-size: 2em;
            page-break-after: avoid;
        }}
        
        h3 {{
            color: #34495e;
            margin-top: 30px;
            margin-bottom: 15px;
            font-size: 1.5em;
            page-break-after: avoid;
        }}
        
        h4 {{
            color: #7f8c8d;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 25px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            page-break-inside: avoid;
        }}
        
        th, td {{
            border: 1px solid #bdc3c7;
            padding: 14px;
            text-align: left;
        }}
        
        th {{
            background: linear-gradient(135deg, #1f77b4 0%, #2980b9 100%);
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.9em;
            letter-spacing: 0.5px;
        }}
        
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        
        tr:hover {{
            background-color: #e8f4f8;
        }}
        
        code {{
            background-color: #ecf0f1;
            padding: 3px 8px;
            border-radius: 4px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.9em;
            color: #e74c3c;
        }}
        
        pre {{
            background-color: #2c3e50;
            color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            overflow-x: auto;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            page-break-inside: avoid;
        }}
        
        pre code {{
            background: none;
            color: #ecf0f1;
            padding: 0;
        }}
        
        blockquote {{
            border-left: 5px solid #3498db;
            margin: 25px 0;
            padding-left: 25px;
            color: #7f8c8d;
            font-style: italic;
            background-color: #f8f9fa;
            padding: 15px 15px 15px 25px;
            border-radius: 0 5px 5px 0;
        }}
        
        ul, ol {{
            margin-left: 25px;
            margin-bottom: 20px;
        }}
        
        li {{
            margin-bottom: 8px;
        }}
        
        strong {{
            color: #2c3e50;
            font-weight: 600;
        }}
        
        hr {{
            border: none;
            border-top: 2px solid #ecf0f1;
            margin: 40px 0;
        }}
        
        a {{
            color: #3498db;
            text-decoration: none;
        }}
        
        a:hover {{
            text-decoration: underline;
        }}
        
        .header-info {{
            text-align: center;
            margin-bottom: 40px;
            padding: 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        
        .footer-info {{
            text-align: center;
            margin-top: 50px;
            padding: 20px;
            border-top: 2px solid #ecf0f1;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        
        @media print {{
            body {{
                padding: 0;
            }}
            .no-print {{
                display: none !important;
            }}
        }}
    </style>
</head>
<body>
    <div class="header-info">
        <h1 style="margin: 0; border: none; color: white;">🛡️ Siber Saldırı Tespit Sistemi</h1>
        <p style="margin: 10px 0 0 0; font-size: 1.2em;">Final Proje Raporu</p>
    </div>
    
    <div class="no-print" style="background: #fff3cd; padding: 15px; border-radius: 5px; margin-bottom: 30px; border-left: 4px solid #ffc107;">
        <strong>💡 PDF'e Dönüştürmek İçin:</strong><br>
        1. <kbd>Ctrl + P</kbd> tuşlarına basın<br>
        2. Yazıcı olarak "<strong>Microsoft Print to PDF</strong>" veya "<strong>PDF olarak kaydet</strong>" seçin<br>
        3. "<strong>Yazdır</strong>" butonuna tıklayın<br>
        4. Dosya adı girin ve kaydedin
    </div>
    
    {html_content}
    
    <div class="footer-info">
        <p><strong>Proje:</strong> Çok-Dataset Ensemble Siber Saldırı Tespit Sistemi</p>
        <p><strong>Geliştirici:</strong> Nefise | <strong>Tarih:</strong> Ocak 2026</p>
        <p><strong>Durum:</strong> ✅ Tamamlandı</p>
    </div>
</body>
</html>
"""

# HTML dosyasını kaydet
output_file = 'FINAL_PROJECT_REPORT.html'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(html_with_style)

print(f"\n✅ HTML dosyası oluşturuldu!")
print(f"📁 Dosya: {output_file}")
print(f"📍 Konum: {os.path.abspath(output_file)}")
print(f"\n🌐 Tarayıcıda açılıyor...")

# Tarayıcıda aç
webbrowser.open(f'file://{os.path.abspath(output_file)}')

print("\n" + "="*70)
print("📄 PDF'E DÖNÜŞTÜRMEK İÇİN:")
print("="*70)
print("1. Tarayıcıda açılan sayfada Ctrl + P tuşlarına basın")
print("2. 'Microsoft Print to PDF' veya 'PDF olarak kaydet' seçin")
print("3. 'Yazdır' butonuna tıklayın")
print("4. Dosya adı girin (örn: Final_Rapor.pdf) ve kaydedin")
print("="*70)
