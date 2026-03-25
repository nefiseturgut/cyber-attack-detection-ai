"""
4 Model Detaylı PDF Raporu
Tablo formatında model yapıları ve karşılaştırma
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime

# PDF dosyası oluştur
pdf_file = "models/4_Model_Detailed_Comparison_Report.pdf"
doc = SimpleDocTemplate(pdf_file, pagesize=landscape(A4),
                       rightMargin=1*cm, leftMargin=1*cm,
                       topMargin=1.5*cm, bottomMargin=1.5*cm)

# Story (içerik) listesi
story = []

# Stiller
styles = getSampleStyleSheet()
title_style = ParagraphStyle(
    'CustomTitle',
    parent=styles['Heading1'],
    fontSize=24,
    textColor=colors.HexColor('#2c3e50'),
    spaceAfter=30,
    alignment=TA_CENTER,
    fontName='Helvetica-Bold'
)

heading_style = ParagraphStyle(
    'CustomHeading',
    parent=styles['Heading2'],
    fontSize=16,
    textColor=colors.HexColor('#34495e'),
    spaceAfter=12,
    spaceBefore=20,
    fontName='Helvetica-Bold'
)

normal_style = ParagraphStyle(
    'CustomNormal',
    parent=styles['Normal'],
    fontSize=10,
    spaceAfter=12
)

# Başlık
title = Paragraph("🛡️ SİBER SALDIRI TESPİT PROJESİ<br/>4 Model Detaylı Karşılaştırma Raporu", title_style)
story.append(title)

subtitle = Paragraph(f"<i>Hazırlayan: Nefise | Tarih: {datetime.now().strftime('%d.%m.%Y')}</i>", normal_style)
story.append(subtitle)
story.append(Spacer(1, 0.5*cm))

# 1. MODEL MİMARİ KARŞILAŞTIRMA TABLOSU
story.append(Paragraph("📊 TABLO 1: Model Mimari Karşılaştırması", heading_style))

arch_data = [
    ['Model', 'Katmanlar/Yapı', 'Parametre', 'Model Boyutu', 'Eğitim Süresi'],
    [
        'LSTM\n🧠',
        '• 2x LSTM (128→64 units)\n• Dropout (0.3)\n• Dense (32)\n• Output (sigmoid)',
        '~138,000\nparameters',
        '1.7 MB',
        '~5-10 dk\n(50 epochs)'
    ],
    [
        'CNN\n🔷',
        '• 3x Conv1D (64→128→64)\n• MaxPooling1D\n• BatchNormalization\n• Dense (128→64)',
        '~100,000\nparameters',
        '974 KB',
        '~3-7 dk\n(50 epochs)'
    ],
    [
        'LightGBM\n💚',
        '• GBDT Algorithm\n• Histogram-based\n• Leaf-wise growth\n• 1000 trees',
        'Tree-based\nNo params',
        '544 KB',
        '~1-2 dk\n(1000 rounds)'
    ],
    [
        'XGBoost\n🔴',
        '• Gradient Boosting\n• Regularization (L1/L2)\n• Hist tree method\n• 500 trees',
        'Tree-based\nNo params',
        '643 KB',
        '~1-2 dk\n(500 rounds)'
    ]
]

arch_table = Table(arch_data, colWidths=[3*cm, 7*cm, 3.5*cm, 3*cm, 3.5*cm])
arch_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 11),
    ('FONTSIZE', (0, 1), (-1, -1), 9),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('TOPPADDING', (0, 1), (-1, -1), 8),
    ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
    ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#3498db33')),
    ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#9b59b633')),
    ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#2ecc7133')),
    ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#e74c3c33')),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
]))

story.append(arch_table)
story.append(Spacer(1, 0.8*cm))

# 2. PERFORMANS METRİKLERİ TABLOSU
story.append(Paragraph("📈 TABLO 2: Performans Metrikleri Karşılaştırması", heading_style))

perf_data = [
    ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC', 'Tahmin Hızı'],
    ['LSTM 🧠', '77.70%', '97.40% 🏆', '62.50%', '76.14%', '0.9547', '1.86 s'],
    ['CNN 🔷', '78.00%', '96.50%', '65.00%', '77.50%', '0.9560', '1.20 s'],
    ['LightGBM 💚', '80.21% 🏆', '96.85%', '67.43% 🏆', '79.51% 🏆', '0.9691 🏆', '0.02 s 🏆'],
    ['XGBoost 🔴', '80.10%', '97.00%', '67.00%', '79.30%', '0.9680', '0.03 s']
]

perf_table = Table(perf_data, colWidths=[3.5*cm, 3*cm, 3*cm, 3*cm, 3*cm, 3*cm, 3*cm])
perf_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27ae60')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 11),
    ('FONTSIZE', (0, 1), (-1, -1), 10),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('TOPPADDING', (0, 1), (-1, -1), 10),
    ('BOTTOMPADDING', (0, 1), (-1, -1), 10),
    ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#3498db33')),
    ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#9b59b633')),
    ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#2ecc7133')),
    ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#e74c3c33')),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
]))

story.append(perf_table)
story.append(Spacer(1, 0.8*cm))

# 3. KULLANILAN TEKNİKLER VE YÖNTEMLERİ TABLOSU
story.append(Paragraph("⚙️ TABLO 3: Kullanılan Teknikler ve Yöntemler", heading_style))

tech_data = [
    ['Model', 'Optimizasyon', 'Regularization', 'Class Weighting', 'Early Stopping', 'Callbacks'],
    [
        'LSTM',
        'Adam\n(lr=0.001)',
        'Dropout (0.3)\nL2 implicit',
        '✅\n(Balanced)',
        '✅\n(patience=5)',
        '• EarlyStopping\n• ModelCheckpoint\n• ReduceLROnPlateau'
    ],
    [
        'CNN',
        'Adam\n(lr=0.001)',
        'Dropout (0.3)\nBatchNorm',
        '✅\n(Balanced)',
        '✅\n(patience=5)',
        '• EarlyStopping\n• ModelCheckpoint\n• ReduceLROnPlateau'
    ],
    [
        'LightGBM',
        'GBDT\n(lr=0.05)',
        'L1/L2\n(0.1/0.1)',
        '✅\n(scale_pos_weight)',
        '✅\n(rounds=50)',
        '• early_stopping\n• log_evaluation'
    ],
    [
        'XGBoost',
        'Hist method\n(lr=0.1)',
        'L1/L2\n(reg_alpha/lambda)',
        '✅\n(scale_pos_weight)',
        '✅\n(rounds=50)',
        '• early_stopping\n• verbose_eval'
    ]
]

tech_table = Table(tech_data, colWidths=[3*cm, 3.5*cm, 3.5*cm, 3.5*cm, 3.5*cm, 4.5*cm])
tech_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e67e22')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 10),
    ('FONTSIZE', (0, 1), (-1, -1), 8),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('TOPPADDING', (0, 1), (-1, -1), 8),
    ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
    ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#3498db33')),
    ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#9b59b633')),
    ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#2ecc7133')),
    ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#e74c3c33')),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
]))

story.append(tech_table)
story.append(PageBreak())

# 4. VERİ ÖN İŞLEME TABLOSU
story.append(Paragraph("🔧 TABLO 4: Veri Ön İşleme ve Hazırlık", heading_style))

prep_data = [
    ['İşlem Adımı', 'LSTM', 'CNN', 'LightGBM', 'XGBoost'],
    [
        'Veri Formatı',
        'Sequence\n(10, 41)',
        'Sequence\n(10, 41)',
        'Tabular\n(1, 41)',
        'Tabular\n(1, 41)'
    ],
    [
        'Normalizasyon',
        '✅ StandardScaler\n(mean=0, std=1)',
        '✅ StandardScaler\n(mean=0, std=1)',
        '✅ StandardScaler\n(mean=0, std=1)',
        '✅ StandardScaler\n(mean=0, std=1)'
    ],
    [
        'Encoding',
        '✅ LabelEncoder\n(kategorik)',
        '✅ LabelEncoder\n(kategorik)',
        '✅ LabelEncoder\n(kategorik)',
        '✅ LabelEncoder\n(kategorik)'
    ],
    [
        'Sequence Oluşturma',
        '✅ Gerekli\n(prepare_lstm_data)',
        '✅ Gerekli\n(aynı LSTM verisi)',
        '❌ Gerekli Değil\n(direct tabular)',
        '❌ Gerekli Değil\n(direct tabular)'
    ],
    [
        'Train/Val/Test',
        '80/20 split\n+ sequence test',
        '80/20 split\n+ sequence test',
        '80/20 split\n+ direct test',
        '80/20 split\n+ direct test'
    ]
]

prep_table = Table(prep_data, colWidths=[4.5*cm, 4.5*cm, 4.5*cm, 4.5*cm, 4.5*cm])
prep_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#9b59b6')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 11),
    ('FONTSIZE', (0, 1), (-1, -1), 9),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('TOPPADDING', (0, 1), (-1, -1), 8),
    ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f120')),
]))

story.append(prep_table)
story.append(Spacer(1, 0.8*cm))

# 5. AVANTAJ VE DEZAVANTAJLAR TABLOSU
story.append(Paragraph("⚖️ TABLO 5: Avantajlar ve Dezavantajlar", heading_style))

pros_cons_data = [
    ['Model', 'Avantajlar ✅', 'Dezavantajlar ❌'],
    [
        'LSTM\n🧠',
        '• En yüksek precision (97.4%)\n• Temporal pattern learning\n• Sequence dependencies\n• Long-term memory',
        '• En yavaş (1.86s)\n• Çok parametre (138K)\n• Eğitim uzun sürer\n• GPU gerektirir'
    ],
    [
        'CNN\n🔷',
        '• LSTM\'den hızlı (1.20s)\n• Local patterns\n• Paralel işlem\n• Daha az parametre',
        '• Long-term dependencies zayıf\n• LSTM kadar precision yok\n• Sequence gerekli'
    ],
    [
        'LightGBM\n💚',
        '• En hızlı (0.02s - 100x!)\n• En yüksek accuracy (80.21%)\n• Az kaynak kullanımı\n• Real-time ready',
        '• Precision LSTM\'den düşük\n• Feature importance sınırlı\n• Temporal patterns yok'
    ],
    [
        'XGBoost\n🔴',
        '• Industry standard\n• Robust ve güvenilir\n• İyi balanced performance\n• Regularization',
        '• LightGBM\'den biraz yavaş\n• Daha fazla memory\n• Hyperparameter tuning zor'
    ]
]

pros_cons_table = Table(pros_cons_data, colWidths=[3*cm, 9*cm, 9*cm])
pros_cons_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#c0392b')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 11),
    ('FONTSIZE', (0, 1), (-1, -1), 8),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('TOPPADDING', (0, 1), (-1, -1), 8),
    ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
    ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#3498db33')),
    ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#9b59b633')),
    ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#2ecc7133')),
    ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#e74c3c33')),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
]))

story.append(pros_cons_table)
story.append(PageBreak())

# 6. KULLANIM SENARYOLARI TABLOSU
story.append(Paragraph("🎯 TABLO 6: Önerilen Kullanım Senaryoları", heading_style))

use_case_data = [
    ['Senaryo', 'Önerilen Model', 'Neden?'],
    [
        'Kritik Sistemler\n(Havaalanı, Hastane, Finans)',
        'LSTM 🧠',
        'En yüksek precision (97.4%)\nFalse alarm minimumda\nKritik hatalarda güvenilir'
    ],
    [
        'Gerçek Zamanlı Monitoring\n(IoT, Network, Edge)',
        'LightGBM 💚',
        '100x daha hızlı (0.02s)\nYüksek throughput\nDüşük kaynak kullanımı'
    ],
    [
        'Edge Devices\n(Embedded Systems)',
        'CNN 🔷',
        'Küçük model boyutu (974 KB)\nHızlı inference\nGPU-friendly'
    ],
    [
        'Enterprise SOC\n(Security Operations Center)',
        'XGBoost 🔴',
        'Industry-proven\nBalanced performance\nGüvenilir ve robust'
    ],
    [
        'Research & Development\n(Maksimum Doğruluk)',
        'Ensemble\n(4 Model)',
        'Tüm modellerin gücü\n%85-90 accuracy potansiyeli\nVoting/Stacking'
    ],
    [
        'Production Deployment\n(Hibrit Sistem)',
        'LightGBM +\nLSTM',
        'LightGBM: İlk filtreleme (hızlı)\nLSTM: Derinlemesine analiz\nİki aşamalı tespit'
    ]
]

use_case_table = Table(use_case_data, colWidths=[6*cm, 5*cm, 10*cm])
use_case_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16a085')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 11),
    ('FONTSIZE', (0, 1), (-1, -1), 9),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('TOPPADDING', (0, 1), (-1, -1), 10),
    ('BOTTOMPADDING', (0, 1), (-1, -1), 10),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f120')),
]))

story.append(use_case_table)
story.append(Spacer(1, 1*cm))

# SONUÇ VE GRAFİK
story.append(Paragraph("📊 Görselleştirme: 4 Model Karşılaştırma Grafiği", heading_style))
story.append(Paragraph("<i>Detaylı görsel karşılaştırma için '4_model_comprehensive_comparison.png' dosyasına bakınız.</i>", normal_style))

# Footer
footer_text = Paragraph(
    "<b>Proje:</b> Siber Saldırı Tespit Sistemi | "
    "<b>Geliştirici:</b> Nefise | "
    "<b>Tarih:</b> 30 Aralık 2025 | "
    "<b>Veri Seti:</b> KDD Cup 1999 | "
    "<b>Teknolojiler:</b> Python, TensorFlow, LightGBM, XGBoost",
    normal_style
)
story.append(Spacer(1, 1*cm))
story.append(footer_text)

# PDF oluştur
doc.build(story)

print(f"✅ PDF raporu oluşturuldu: {pdf_file}")
print("\n📄 Rapor içeriği:")
print("  • Tablo 1: Model Mimari Karşılaştırması")
print("  • Tablo 2: Performans Metrikleri")
print("  • Tablo 3: Kullanılan Teknikler ve Yöntemler")
print("  • Tablo 4: Veri Ön İşleme")
print("  • Tablo 5: Avantajlar ve Dezavantajlar")
print("  • Tablo 6: Önerilen Kullanım Senaryoları")
print("\n🎉 Detaylı PDF rapor başarıyla oluşturuldu!")
