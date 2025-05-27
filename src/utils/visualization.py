import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal
from utils.audio_io import load_audio

def plot_waveform_comparison(clean, noisy, filtered, sr, title="Waveform Comparison"):
    """
    Zaman domeninde sinyalleri karşılaştırmalı gösterir
    
    :param clean: Temiz ses sinyali
    :param noisy: Gürültülü ses sinyali
    :param filtered: Filtrelenmiş ses sinyali
    :param sr: Örnekleme oranı
    :param title: Grafik başlığı
    """
    plt.figure(figsize=(12, 8))
    
    # Zaman eksenini oluştur
    time = np.arange(len(clean)) / sr
    
    # 3 alt grafik oluştur
    plt.subplot(3, 1, 1)
    plt.plot(time, clean, color='blue', alpha=0.7)
    plt.title('Original Clean Audio')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 2)
    plt.plot(time, noisy, color='red', alpha=0.7)
    plt.title('Noisy Audio')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 3)
    plt.plot(time, filtered, color='green', alpha=0.7)
    plt.title('Filtered Audio')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Grafiği kaydet
    plot_path = os.path.join('results', 'plots', 'waveform_comparison.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_spectrogram(audio_data, sr, title="Spectrogram", cmap='viridis'):
    """
    Ses sinyalinin spektrogramını çizer
    
    :param audio_data: Ses sinyali
    :param sr: Örnekleme oranı
    :param title: Grafik başlığı
    :param cmap: Renk haritası
    """
    plt.figure(figsize=(12, 4))
    
    # Spektrogram hesapla
    f, t, Sxx = signal.spectrogram(audio_data, fs=sr, nperseg=512, noverlap=256)
    
    # dB cinsine çevir
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    # Spektrogramı çiz
    plt.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap=cmap)
    plt.colorbar(label='Intensity (dB)')
    plt.title(title)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.ylim(0, sr//2)  # Nyquist frekansına kadar göster
    
    # Grafiği kaydet
    plot_path = os.path.join('results', 'plots', f'spectrogram_{title.lower().replace(" ", "_")}.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_frequency_response(filter_coeffs, sr, title="Filter Frequency Response"):
    """
    Filtrenin frekans tepkisini çizer
    
    :param filter_coeffs: Filtre katsayıları
    :param sr: Örnekleme oranı
    :param title: Grafik başlığı
    """
    plt.figure(figsize=(10, 5))
    
    # Frekans tepkisini hesapla
    w, h = signal.freqz(filter_coeffs, fs=sr)
    
    # Genlik tepkisi (dB cinsinden)
    plt.subplot(2, 1, 1)
    plt.plot(w, 20 * np.log10(np.abs(h) + 1e-10))
    plt.title(title)
    plt.ylabel('Amplitude (dB)')
    plt.grid(True)
    
    # Faz tepkisi
    plt.subplot(2, 1, 2)
    plt.plot(w, np.unwrap(np.angle(h)))
    plt.ylabel('Phase (radians)')
    plt.xlabel('Frequency (Hz)')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Grafiği kaydet
    plot_path = os.path.join('results', 'plots', 'filter_response.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_comprehensive_analysis(clean, noisy, filtered, sr, noise_bands=None):
    """
    Kapsamlı bir analiz paneli oluşturur
    
    :param clean: Temiz ses sinyali
    :param noisy: Gürültülü ses sinyali
    :param filtered: Filtrelenmiş ses sinyali
    :param sr: Örnekleme oranı
    :param noise_bands: Gürültü bantları [(start1, end1), ...]
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig)
    
    # 1. Zaman domeni karşılaştırması
    ax1 = fig.add_subplot(gs[0, :])
    time = np.arange(len(clean)) / sr
    ax1.plot(time, clean, label='Clean', alpha=0.7)
    ax1.plot(time, noisy, label='Noisy', alpha=0.5)
    ax1.plot(time, filtered, label='Filtered', alpha=0.7)
    ax1.set_title('Time Domain Comparison')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Frekans spektrumu
    ax2 = fig.add_subplot(gs[1, 0])
    freqs = np.fft.rfftfreq(len(clean), 1/sr)
    clean_fft = np.abs(np.fft.rfft(clean))
    noisy_fft = np.abs(np.fft.rfft(noisy))
    filtered_fft = np.abs(np.fft.rfft(filtered))
    
    ax2.semilogx(freqs, 20*np.log10(clean_fft+1e-10), label='Clean')
    ax2.semilogx(freqs, 20*np.log10(noisy_fft+1e-10), label='Noisy', alpha=0.6)
    ax2.semilogx(freqs, 20*np.log10(filtered_fft+1e-10), label='Filtered')
    
    # Gürültü bantlarını işaretle
    if noise_bands:
        for band in noise_bands:
            ax2.axvspan(band[0], band[1], color='red', alpha=0.1)
    
    ax2.set_title('Frequency Spectrum')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude (dB)')
    ax2.legend()
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    
    # 3. Spektrogram karşılaştırması
    ax3 = fig.add_subplot(gs[1, 1])
    f, t, Sxx = signal.spectrogram(filtered, fs=sr, nperseg=512, noverlap=256)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    im = ax3.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis')
    fig.colorbar(im, ax=ax3, label='Intensity (dB)')
    ax3.set_title('Filtered Audio Spectrogram')
    ax3.set_ylabel('Frequency (Hz)')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylim(0, sr//4)  # Daha iyi görünüm için
    
    # 4. SNR zaman serisi
    ax4 = fig.add_subplot(gs[2, 0])
    window_size = int(0.05 * sr)  # 50ms pencereler
    snr_values = []
    for i in range(0, len(clean), window_size):
        chunk_clean = clean[i:i+window_size]
        chunk_filtered = filtered[i:i+window_size]
        snr = 10 * np.log10(np.sum(chunk_clean**2) / (np.sum((chunk_filtered - chunk_clean)**2) + 1e-10))
        snr_values.append(snr)
    
    ax4.plot(np.arange(len(snr_values)) * window_size/sr, snr_values)
    ax4.set_title('Local SNR Over Time')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('SNR (dB)')
    ax4.grid(True)
    
    # 5. Histogram karşılaştırması
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.hist(clean, bins=100, alpha=0.5, density=True, label='Clean')
    ax5.hist(filtered, bins=100, alpha=0.5, density=True, label='Filtered')
    ax5.set_title('Amplitude Distribution')
    ax5.set_xlabel('Amplitude')
    ax5.set_ylabel('Density')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Grafiği kaydet
    plot_path = os.path.join('results', 'plots', 'comprehensive_analysis.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
