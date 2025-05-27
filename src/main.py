import os
import numpy as np
import matplotlib.pyplot as plt
from utils.audio_io import load_audio, save_audio
from analysis.frequency_analysis import analyze_audio_pair
from filters.fir_filter import FIRFilter
from filters.iir_filter import IIRFilter
from filters.band_stop_filter import BandStopFilter
from utils.metrics import calculate_snr, plot_quality_metrics
from utils.visualization import plot_comprehensive_analysis
from analysis.time_analysis import plot_energy_comparison, plot_zcr_comparison

def process_audio_pair(clean_path, noisy_path, output_dir):
    """
    Temiz ve gürültülü ses çiftini işler
    
    :param clean_path: Temiz ses dosyası yolu
    :param noisy_path: Gürültülü ses dosyası yolu
    :param output_dir: Çıktı dizini
    """
    # 1. Ses dosyalarını yükle
    clean_audio, sr = load_audio(clean_path)
    noisy_audio, _ = load_audio(noisy_path)
    
    # 2. Frekans analizi yap ve gürültü bantlarını tespit et
    print(f"\nFrekans analizi yapılıyor: {os.path.basename(clean_path)}")
    noise_bands = analyze_audio_pair(clean_path, noisy_path)
    print(f"Tespit edilen gürültü bantları: {noise_bands} Hz")
    
    # 3. Filtre tasarla ve uygula (Birden fazla filtre deneyebilirsiniz)
    print("\nFiltreleme işlemi başlatılıyor...")
    
    # Seçenek 1: Özel bant durdurma filtresi
    bsf = BandStopFilter(sr, noise_bands, filter_type='fir', order=201)
    filtered_audio = bsf.apply(noisy_audio)
    
    # Seçenek 2: FIR band-stop filtresi (alternatif)
    # fir_filter = FIRFilter(sr, noise_bands, filter_type='bandstop', order=251)
    # filtered_audio = fir_filter.apply(noisy_audio)
    
    # Seçenek 3: IIR band-stop filtresi (alternatif)
    # iir_filter = IIRFilter(sr, noise_bands, filter_type='bandstop', iir_type='ellip', order=6, rp=1, rs=60)
    # filtered_audio = iir_filter.apply(noisy_audio)
    
    # 4. Sonuçları kaydet
    os.makedirs(output_dir, exist_ok=True)
    
    # Filtrelenmiş sesi kaydet
    output_audio_path = os.path.join(output_dir, 'filtered_' + os.path.basename(noisy_path))
    save_audio(output_audio_path, filtered_audio, sr)
    
    # 5. Performans metriklerini hesapla
    print("\nPerformans metrikleri hesaplanıyor...")
    snr_before = calculate_snr(clean_audio, noisy_audio)
    snr_after = calculate_snr(clean_audio, filtered_audio)
    print(f"SNR İyileşmesi: {snr_after-snr_before:.2f} dB (Önce: {snr_before:.2f} dB, Sonra: {snr_after:.2f} dB)")
    
    # 6. Görselleştirmeleri oluştur
    print("\nGörselleştirmeler oluşturuluyor...")
    
    # Kapsamlı analiz grafiği
    plot_comprehensive_analysis(clean_audio, noisy_audio, filtered_audio, sr, noise_bands)
    
    # Kalite metrikleri grafiği
    plot_quality_metrics(clean_audio, noisy_audio, filtered_audio, sr)
    
    # Zaman domeni analizleri
    plot_energy_comparison(clean_audio, noisy_audio, filtered_audio, sr)
    plot_zcr_comparison(clean_audio, noisy_audio, filtered_audio, sr)
    
    # Filtre tepkisini kaydet
    bsf.plot_response(os.path.join(output_dir, 'filter_response.png'))
    
    print(f"\nİşlem tamamlandı. Sonuçlar '{output_dir}' klasörüne kaydedildi.")

def batch_process(data_dir, output_base_dir):
    """
    Tüm ses dosyalarını toplu işleme tabi tutar
    
    :param data_dir: Veri klasörü yolu (içinde original ve noisy klasörleri olmalı)
    :param output_base_dir: Çıktıların kaydedileceği temel dizin
    """
    original_dir = os.path.join(data_dir, 'original')
    noisy_dir = os.path.join(data_dir, 'noisy')
    
    # Tüm ses dosyalarını işle
    for file in os.listdir(original_dir):
        if file.endswith('.wav'):
            clean_path = os.path.join(original_dir, file)
            noisy_path = os.path.join(noisy_dir, file)
            
            # Her dosya için ayrı çıktı dizini oluştur
            output_dir = os.path.join(output_base_dir, os.path.splitext(file)[0])
            
            print(f"\n{'='*50}")
            print(f"{file} dosyası işleniyor...")
            print(f"{'='*50}")
            
            try:
                process_audio_pair(clean_path, noisy_path, output_dir)
            except Exception as e:
                print(f"{file} işlenirken hata oluştu: {str(e)}")
                continue

if __name__ == "__main__":
    # Ana yürütme kodu
    DATA_DIR = 'data'  # original ve noisy klasörlerini içeren dizin
    OUTPUT_DIR = 'results'  # Tüm sonuçların kaydedileceği dizin
    
    print("""
    ###################################################
    #       İşaretler ve Sistemler Projesi            #
    #       Gürültülü Ses Filtreleme Sistemi          #
    ###################################################
    """)
    
    # Kullanıcı menüsü
    while True:
        print("\nMenü:")
        print("1. Tek dosya işle")
        print("2. Tüm dosyaları toplu işle")
        print("3. Çıkış")
        
        choice = input("Seçiminiz (1-3): ")
        
        if choice == '1':
            # Tek dosya işleme
            file_name = input("İşlenecek dosya adı (örn: sample1.wav): ")
            clean_path = os.path.join(DATA_DIR, 'original', file_name)
            noisy_path = os.path.join(DATA_DIR, 'noisy', file_name)
            
            if os.path.exists(clean_path) and os.path.exists(noisy_path):
                output_dir = os.path.join(OUTPUT_DIR, os.path.splitext(file_name)[0])
                process_audio_pair(clean_path, noisy_path, output_dir)
            else:
                print("Hata: Dosyalar bulunamadı. Lütfen kontrol edin.")
        
        elif choice == '2':
            # Toplu işleme
            print("\nToplu işlem başlatılıyor...")
            batch_process(DATA_DIR, OUTPUT_DIR)
            print("\nToplu işlem tamamlandı.")
        
        elif choice == '3':
            print("Program sonlandırılıyor...")
            break
        
        else:
            print("Geçersiz seçim. Lütfen 1-3 arasında bir sayı girin.")