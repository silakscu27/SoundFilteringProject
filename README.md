# SoundFilteringProject

# 🎧 Ses Gürültüsü Filtreleme Projesi – İşaretler ve Sistemler (Bahar 2024)

Bu proje, Sakarya Uygulamalı Bilimler Üniversitesi "İşaretler ve Sistemler" dersi kapsamında geliştirilmiştir. Amaç; gürültülü ses kayıtlarından istenmeyen frekans bileşenlerini **klasik sinyal işleme teknikleri** kullanarak filtrelemektir.

## 📌 Proje Amacı

Verilen veri setinde, temiz ve gürültülü ses kayıtları eşlenmiş olarak sunulmuştur. Projede bu gürültülü seslerin frekans içeriği analiz edilecek, **band-stop** gibi uygun filtreler tasarlanacak ve bu filtreler sinyallere uygulanarak gürültü bastırılacaktır. Son olarak, elde edilen sonuçlar hem **sayısal metrikler** (SNR) ile hem de **görsel analizlerle** değerlendirilecektir.

## 🧠 Kısıtlamalar ve Kurallar

✅ Kullanılabilir:
- Klasik sinyal işleme teknikleri (FFT, IIR/FIR filtreler)
- Python kütüphaneleri: `numpy`, `scipy`, `matplotlib`, `librosa`, `soundfile`

❌ Kullanılamaz:
- Makine öğrenmesi yöntemleri
- Hazır gürültü azaltma fonksiyonları (örneğin `noisereduce.reduce_noise` gibi)
- Otomatik filtreleme kütüphaneleri (deep learning tabanlı)

