import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def calculate_snr(clean_signal, noisy_signal):
    """
    Sinyal-Gürültü Oranı (SNR) hesaplar (dB cinsinden)
    
    :param clean_signal: Temiz ses sinyali (numpy array)
    :param noisy_signal: Gürültülü ses sinyali (numpy array)
    :return: SNR değeri (dB)
    """
    # Sinyal ve gürültü enerjilerini hesapla
    signal_energy = np.sum(clean_signal**2)
    noise_energy = np.sum((noisy_signal - clean_signal)**2)
    
    # Sıfıra bölme hatasını önle
    if noise_energy < 1e-10:
        return float('inf')
    
    snr = 10 * np.log10(signal_energy / noise_energy)
    return snr

def calculate_mse(original, processed):
    """
    Ortalama Kare Hata (MSE) hesaplar
    
    :param original: Orijinal sinyal
    :param processed: İşlenmiş sinyal
    :return: MSE değeri
    """
    return np.mean((original - processed)**2)

def calculate_psnr(original, processed, max_val=1.0):
    """
    Tepe Sinyal-Gürültü Oranı (PSNR) hesaplar (dB cinsinden)
    
    :param original: Orijinal sinyal
    :param processed: İşlenmiş sinyal
    :param max_val: Maksimum olası sinyal değeri
    :return: PSNR değeri (dB)
    """
    mse = calculate_mse(original, processed)
    if mse < 1e-10:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))

def spectral_distance(original, processed, sr):
    """
    Spektral mesafeyi hesaplar (log spektral mesafe)
    
    :param original: Orijinal sinyal
    :param processed: İşlenmiş sinyal
    :param sr: Örnekleme oranı
    :return: Log spektral mesafe
    """
    # FFT hesapla
    orig_fft = np.abs(np.fft.fft(original)[:len(original)//2])
    proc_fft = np.abs(np.fft.fft(processed)[:len(processed)//2])
    
    # Güç spektrumlarını hesapla
    orig_power = 10 * np.log10(orig_fft**2 + 1e-10)
    proc_power = 10 * np.log10(proc_fft**2 + 1e-10)
    
    # Log spektral mesafe
    lsd = np.sqrt(np.mean((orig_power - proc_power)**2))
    return lsd

def plot_quality_metrics(original, noisy, filtered, sr, title="Quality Metrics"):
    """
    Ses kalite metriklerini görselleştirir
    
    :param original: Orijinal temiz sinyal
    :param noisy: Gürültülü sinyal
    :param filtered: Filtrelenmiş sinyal
    :param sr: Örnekleme oranı
    :param title: Grafik başlığı
    """
    # Metrikleri hesapla
    snr_noisy = calculate_snr(original, noisy)
    snr_filtered = calculate_snr(original, filtered)
    mse_noisy = calculate_mse(original, noisy)
    mse_filtered = calculate_mse(original, filtered)
    lsd_noisy = spectral_distance(original, noisy, sr)
    lsd_filtered = spectral_distance(original, filtered, sr)
    
    # Grafik oluştur
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # SNR Karşılaştırması
    ax[0].bar(['Noisy', 'Filtered'], [snr_noisy, snr_filtered], color=['red', 'green'])
    ax[0].set_title('SNR (dB) Comparison')
    ax[0].set_ylabel('dB')
    
    # MSE Karşılaştırması
    ax[1].bar(['Noisy', 'Filtered'], [mse_noisy, mse_filtered], color=['red', 'green'])
    ax[1].set_title('MSE Comparison')
    ax[1].set_yscale('log')
    
    # LSD Karşılaştırması
    ax[2].bar(['Noisy', 'Filtered'], [lsd_noisy, lsd_filtered], color=['red', 'green'])
    ax[2].set_title('Spectral Distance')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Grafiği kaydet
    plot_path = os.path.join('results', 'plots', 'quality_metrics.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
