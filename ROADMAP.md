# 🚀 Siber Saldırı Tespit Sistemi - İleri Aşama Yol Haritası (Roadmap)

Bu doküman, makine öğrenmesi modelleri geliştirildikten sonra projenin **Gerçek Zamanlı Simülasyon ve İzleme** aşamasına geçişi için hazırlanmış adım adım bir plandır.

---

## 🛠️ Aşama 1: Modellerin İyileştirilmesi ve Optimize Edilmesi

- [ ] **Daha Fazla Epoch ile Eğitim**: Ağır modellerin (LSTM, CNN) iterasyon (epoch) sayılarını artırarak over-fitting (aşırı öğrenme) engellenerek gerçeğe en yakın değerlere ulaşılması.
- [ ] **Hiperparametre Optimizasyonu**: Özellik seçimi (feature selection) ve hiperparametre ayarlamalarıyla modellerin gerçek zamanlı trafikte daha stabil çalışmasının sağlanması.
- [ ] **Model Dışa Aktarımı**: Modellerin canlı dinleme sistemine (monitoring) hızlı cevap verebilmesi için en verimli formatta (.keras, .pkl) pipeline'a hazır hale getirilmesi.

---

## 🌐 Aşama 2: Sanal Ağ (Virtual Network) Topolojisinin Kurulması

- [ ] **Sanal Ağ Ortamının Seçilmesi**: VirtualBox, VMware veya Docker kullanılarak izole bir sanal ağ oluşturulması.
- [ ] **Hedef (Target) ve Saldırgan (Attacker) Makineler**: 
  - Saldırgan Makine (Örn: **Kali Linux**)
  - Hedef Makine (Örn: Ubuntu veya Metasploitable)
- [ ] **Ağ Dinleme (Sniffing) Altyapısı**: Ağ içerisindeki paketleri canlı olarak yakalamak için hedef makineye (veya host makinenize) **Wireshark, TShark veya Scapy/PyShark** tabanlı bir paket toplayıcı entegre edilmesi.

---

## ⚔️ Aşama 3: Saldırı Senaryoları ve Canlı Trafik Analizi

- [ ] **Saldırıların Simüle Edilmesi**: Saldırgan makineden Nmap taramaları, DoS/DDoS (hping3 vb.), kaba kuvvet (brute-force) gibi hareketliliklerin başlatılması.
- [ ] **Gerçek Zamanlı Trafik Dönüşümü**: Yakalanan pcap (paket) dosyalarının canlı olarak Python içerisinde parse edilmesi ve **CICFlowMeter** benzeri bir script kullanılarak modellerin anlayacağı 80/42 (CICIDS/UNSW) sütunlu özniteliklere dönüştürülmesi.
- [ ] **Modellere Besleme (Live Feed)**: Çıkarılan bu özelliklerin anlık olarak Ensemble AI modeline gönderilip tahmin skorlarının alınması.

---

## 🖥️ Aşama 4: Masaüstü Monitöring (İzleme) Uygulaması Geliştirilmesi

- [ ] **Arayüz (GUI) Geliştirme**: 
  - **PyQt5/PySide6** (Özel bir masaüstü uygulaması) veya
  - **Streamlit** (Modern ve şık bir yerel web paneli) kullanılarak gösterge panelinin oluşturulması.
- [ ] **Canlı Metriklerin Gösterimi**:
  - Güvenli vs. Zararlı trafik oranlarını gösteren canlı pasta/çizgi grafikleri.
  - Ağdaki paketleri ve gelen/giden IP'leri gösteren akan bir log (kayıt) ekranı.
  - Siber saldırı tespit edildiğinde **Kırmızı Alarm (Alert)** verilmesi ve saldırı türünün ekrana yansıtılması.

---

## 🏗️ Sistem Mimarisi

```mermaid
graph TD
    subgraph Sanal Ağ (Virtual Network)
        A((Saldırgan Makine<br>Örn: Kali Linux)) --> |DoS, Tarama, Brute-force| B[Hedef Makine]
        NormalUser((Normal Kullanıcı)) --> |Olağan Trafik| B
    end
    B --> C(Ağ Dinleyici Aracımız<br>PyShark / Scapy)
    C --> D[Özellik Çıkarımı<br>Packet to DataFrame]
    
    subgraph Yapay Zeka & İzleme (Monitoring)
        D -->|Ağ Özellikleri| E{Ensemble AI Modeli}
        E -->|Güvenli| F[Masaüstü Monitoring GUI]
        E -->|Anomali Tespit Edildi| F
        F -->|Dashboard & Alarm| G((Siber Güvenlik Uzmanı))
    end
```
