import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from utils.audio_io import load_audio

def compute_fft(audio_data, sample_rate):
    """
    Saf Python/NumPy ile FFT hesaplar
    :param audio_data: Ses sinyalinin zaman domeni verisi
    :param sample_rate: Örnekleme frekansı
    :return: (frekanslar, genlik spektrumu) tuple'ı
    """
    n = len(audio_data)
    fft_result = np.fft.fft(audio_data)
    amplitudes = np.abs(fft_result)[:n//2] * 2/n  # Normalizasyon
    freqs = np.fft.fftfreq(n, 1/sample_rate)[:n//2]
    return freqs, amplitudes

def identify_noise_bands(clean_audio, noisy_audio, sample_rate, threshold_db=20):
    """
    Temiz ve gürültülü sinyaller arasındaki farktan gürültü bantlarını belirler
    :param clean_audio: Temiz ses sinyali
    :param noisy_audio: Gürültülü ses sinyali
    :param sample_rate: Örnekleme frekansı
    :param threshold_db: Gürültü olarak kabul edilecek minimum dB farkı
    :return: Gürültülü frekans bantlarının listesi [(başlangıç_frekansı, bitiş_frekansı)]
    """
    # FFT hesapla
    clean_freqs, clean_amps = compute_fft(clean_audio, sample_rate)
    noisy_freqs, noisy_amps = compute_fft(noisy_audio, sample_rate)
    
    # dB cinsinden hesapla
    clean_db = 20 * np.log10(clean_amps + 1e-10)  # Sıfır bölme hatasını önle
    noisy_db = 20 * np.log10(noisy_amps + 1e-10)
    diff_db = noisy_db - clean_db
    
    # Gürültü bantlarını belirle
    noise_bands = []
    in_noise_band = False
    band_start = 0
    
    for i, (freq, db_diff) in enumerate(zip(clean_freqs, diff_db)):
        if db_diff > threshold_db and not in_noise_band:
            band_start = freq
            in_noise_band = True
        elif db_diff <= threshold_db and in_noise_band:
            noise_bands.append((band_start, freq))
            in_noise_band = False
    
    # Sonlandırılmamış bant varsa ekle
    if in_noise_band:
        noise_bands.append((band_start, clean_freqs[-1]))
    
    return noise_bands

def plot_frequency_comparison(clean_audio, noisy_audio, sample_rate, title="Frequency Spectrum Comparison"):
    """
    Temiz ve gürültülü sinyallerin frekans spektrumunu karşılaştırmalı gösterir
    :param clean_audio: Temiz ses sinyali
    :param noisy_audio: Gürültülü ses sinyali
    :param sample_rate: Örnekleme frekansı
    :param title: Grafik başlığı
    """
    clean_freqs, clean_amps = compute_fft(clean_audio, sample_rate)
    noisy_freqs, noisy_amps = compute_fft(noisy_audio, sample_rate)
    
    plt.figure(figsize=(12, 6))
    plt.semilogx(clean_freqs, 20 * np.log10(clean_amps + 1e-10), label='Clean Audio')
    plt.semilogx(noisy_freqs, 20 * np.log10(noisy_amps + 1e-10), alpha=0.7, label='Noisy Audio')
    
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    
    # Grafiği kaydet
    plot_path = os.path.join('results', 'plots', 'frequency_comparison.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()

def analyze_audio_pair(clean_path, noisy_path):
    """
    Temiz ve gürültülü ses çiftini analiz eder
    :param clean_path: Temiz ses dosyası yolu
    :param noisy_path: Gürültülü ses dosyası yolu
    :return: Gürültü bantları ve analiz grafiği yolu
    """
    clean_audio, sr = load_audio(clean_path)
    noisy_audio, _ = load_audio(noisy_path)
    
    # Frekans analizi yap
    noise_bands = identify_noise_bands(clean_audio, noisy_audio, sr)
    
    # Grafik oluştur
    plot_frequency_comparison(clean_audio, noisy_audio, sr)
    
    return noise_bands

if __name__ == "__main__":
    # Test kodu
    clean_test_path = os.path.join('data', 'original', 'test_clean.wav')
    noisy_test_path = os.path.join('data', 'noisy', 'test_noisy.wav')
    
    bands = analyze_audio_pair(clean_test_path, noisy_test_path)
    print(f"Identified noise bands: {bands}")