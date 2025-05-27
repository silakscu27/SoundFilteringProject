import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
from utils.audio_io import load_audio

def calculate_rms(audio_data, window_size=1024):
    """
    Ses sinyalinin RMS (Root Mean Square) değerini pencere bazında hesaplar
    
    :param audio_data: Ses sinyal verisi
    :param window_size: Pencere boyutu (samples)
    :return: RMS değerleri dizisi
    """
    num_windows = len(audio_data) // window_size
    rms_values = np.zeros(num_windows)
    
    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        window = audio_data[start:end]
        rms_values[i] = np.sqrt(np.mean(window**2))
    
    return rms_values

def calculate_zcr(audio_data, window_size=1024):
    """
    Sıfır Geçiş Oranı (Zero Crossing Rate) hesaplar
    
    :param audio_data: Ses sinyal verisi
    :param window_size: Pencere boyutu (samples)
    :return: ZCR değerleri dizisi
    """
    num_windows = len(audio_data) // window_size
    zcr_values = np.zeros(num_windows)
    
    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        window = audio_data[start:end]
        zcr_values[i] = 0.5 * np.mean(np.abs(np.diff(np.sign(window))))
    
    return zcr_values

def plot_energy_comparison(clean, noisy, filtered, sr, window_size=1024):
    """
    Enerji karşılaştırmasını zaman domeninde gösterir
    
    :param clean: Temiz ses sinyali
    :param noisy: Gürültülü ses sinyali
    :param filtered: Filtrelenmiş ses sinyali
    :param sr: Örnekleme oranı
    :param window_size: Analiz penceresi boyutu
    """
    # RMS değerlerini hesapla
    clean_rms = calculate_rms(clean, window_size)
    noisy_rms = calculate_rms(noisy, window_size)
    filtered_rms = calculate_rms(filtered, window_size)
    
    # Zaman eksenini oluştur
    time = np.arange(len(clean_rms)) * window_size / sr
    
    plt.figure(figsize=(12, 6))
    plt.plot(time, clean_rms, label='Clean Audio', alpha=0.8)
    plt.plot(time, noisy_rms, label='Noisy Audio', alpha=0.6)
    plt.plot(time, filtered_rms, label='Filtered Audio', alpha=0.8)
    
    plt.title('Energy (RMS) Comparison Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('RMS Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Grafiği kaydet
    plot_path = os.path.join('results', 'plots', 'energy_comparison.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_zcr_comparison(clean, noisy, filtered, sr, window_size=1024):
    """
    Sıfır geçiş oranlarını karşılaştırmalı gösterir
    
    :param clean: Temiz ses sinyali
    :param noisy: Gürültülü ses sinyali
    :param filtered: Filtrelenmiş ses sinyali
    :param sr: Örnekleme oranı
    :param window_size: Analiz penceresi boyutu
    """
    # ZCR değerlerini hesapla
    clean_zcr = calculate_zcr(clean, window_size)
    noisy_zcr = calculate_zcr(noisy, window_size)
    filtered_zcr = calculate_zcr(filtered, window_size)
    
    # Zaman eksenini oluştur
    time = np.arange(len(clean_zcr)) * window_size / sr
    
    plt.figure(figsize=(12, 6))
    plt.plot(time, clean_zcr, label='Clean Audio', alpha=0.8)
    plt.plot(time, noisy_zcr, label='Noisy Audio', alpha=0.6)
    plt.plot(time, filtered_zcr, label='Filtered Audio', alpha=0.8)
    
    plt.title('Zero Crossing Rate (ZCR) Comparison Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('ZCR')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Grafiği kaydet
    plot_path = os.path.join('results', 'plots', 'zcr_comparison.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def detect_silence(audio_data, sr, threshold_db=-40, min_silence_duration=0.1):
    """
    Sessiz bölümleri tespit eder
    
    :param audio_data: Ses sinyali
    :param sr: Örnekleme oranı
    :param threshold_db: Sessizlik eşiği (dB)
    :param min_silence_duration: Minimum sessizlik süresi (saniye)
    :return: (start_indices, end_indices) tuple'ı
    """
    # Enerjiyi hesapla (dB cinsinden)
    amplitude = np.abs(audio_data)
    amplitude_db = 20 * np.log10(amplitude + 1e-10)  # Sıfır bölme hatasını önle
    
    # Sessizlik eşiğini uygula
    silent_regions = amplitude_db < threshold_db
    
    # Sessiz bölgeleri bul
    silent_samples = int(min_silence_duration * sr)
    state_changes = np.diff(silent_regions.astype(int))
    starts = np.where(state_changes == 1)[0]
    ends = np.where(state_changes == -1)[0]
    
    # Başlangıç ve bitişleri düzelt
    if len(ends) == 0:
        ends = np.array([len(audio_data)-1])
    
    if len(starts) == 0 or starts[0] > ends[0]:
        starts = np.insert(starts, 0, 0)
    
    if ends[-1] < starts[-1]:
        ends = np.append(ends, len(audio_data)-1)
    
    # Minimum süre koşulunu uygula
    valid_segments = []
    for start, end in zip(starts, ends):
        if (end - start) >= silent_samples:
            valid_segments.append((start, end))
    
    if not valid_segments:
        return np.array([]), np.array([])
    
    return np.array(valid_segments).T

def plot_silence_detection(audio_data, sr, title="Silence Detection"):
    """
    Sessiz bölümleri görselleştirir
    
    :param audio_data: Ses sinyali
    :param sr: Örnekleme oranı
    :param title: Grafik başlığı
    """
    # Sessiz bölümleri tespit et
    starts, ends = detect_silence(audio_data, sr)
    
    plt.figure(figsize=(12, 4))
    time = np.arange(len(audio_data)) / sr
    
    # Ses dalga formunu çiz
    plt.plot(time, audio_data, label='Audio Signal')
    
    # Sessiz bölgeleri işaretle
    for start, end in zip(starts, ends):
        plt.axvspan(time[start], time[end], color='red', alpha=0.3)
    
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    # Grafiği kaydet
    plot_path = os.path.join('results', 'plots', 'silence_detection.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
