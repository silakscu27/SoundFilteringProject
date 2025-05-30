import numpy as np
from scipy.signal import butter, sosfilt, sosfreqz

def design_band_stop_filter(lowcut, highcut, fs, order=4):
    """
    Band-stop (notch) filtre tasarımı. Butterworth tipi.
    Parametreler:
        lowcut: Alt frekans sınırı (Hz)
        highcut: Üst frekans sınırı (Hz)
        fs: Örnekleme frekansı (Hz)
        order: Filtre derecesi
    Geriye:
        sos: Second-order sections (kararlı yapı)
    """
    if not 0 < lowcut < highcut < fs / 2:
        raise ValueError(f"Geçersiz frekans aralığı: {lowcut}-{highcut}Hz, Nyquist = {fs/2}Hz")

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    try:
        sos = butter(order, [low, high], btype='bandstop', output='sos')
    except Exception as e:
        raise RuntimeError(f"Filtre tasarımı başarısız: {e}")
    
    return sos

def apply_filter(signal, sos):
    """
    Verilen sinyale band-stop filtresini uygular.
    Parametreler:
        signal: Giriş sinyali (numpy array)
        sos: Second-order section filtre yapısı
    Geriye:
        filtered_signal: Filtrelenmiş sinyal
    """
    if not isinstance(signal, np.ndarray):
        raise TypeError("Giriş sinyali bir numpy array olmalıdır.")

    try:
        filtered_signal = sosfilt(sos, signal)
    except Exception as e:
        raise RuntimeError(f"Filtre uygulanırken hata oluştu: {e}")

    return filtered_signal
