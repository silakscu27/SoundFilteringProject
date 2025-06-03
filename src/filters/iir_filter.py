import numpy as np
from scipy.signal import butter, filtfilt, lfilter

def design_iir_filter(filter_type, fs, order=4, cutoff=None, band=None):
    """
    IIR filtre tasarımı (Butterworth).

    Parametreler:
        filter_type: 'lowpass', 'highpass', 'bandpass', 'bandstop'
        fs: örnekleme frekansı (Hz)
        order: filtre derecesi
        cutoff: low/high filtreler için frekans değeri (Hz)
        band: bandpass/bandstop için [low, high] (Hz)

    Geriye:
        b, a: Filtre katsayıları
    """
    nyq = fs / 2  # Nyquist frekansı

    if filter_type in ['lowpass', 'highpass']:
        if cutoff is None:
            raise ValueError("cutoff frekansı belirtilmeli.")
        norm_cutoff = cutoff / nyq
        b, a = butter(order, norm_cutoff, btype=filter_type, analog=False)

    elif filter_type in ['bandpass', 'bandstop']:
        if band is None or len(band) != 2:
            raise ValueError("Bandpass/Bandstop için band=[low, high] girilmeli.")
        norm_band = [band[0] / nyq, band[1] / nyq]
        b, a = butter(order, norm_band, btype=filter_type, analog=False)

    else:
        raise ValueError(f"Geçersiz filtre türü: {filter_type}")

    return b, a


def apply_iir_filter(signal, b, a, zero_phase=True):
    """
    IIR filtresini uygular. İsteğe bağlı olarak sıfır faz bozulmalı (filtfilt) çalışır.

    Parametreler:
        signal: giriş sinyali (numpy array)
        b, a: filtre katsayıları
        zero_phase: True ise filtfilt (faz bozmaz), False ise lfilter

    Geriye:
        filtered: filtrelenmiş sinyal
    """
    if not isinstance(signal, np.ndarray):
        raise TypeError("Giriş sinyali numpy array olmalı.")

    try:
        if zero_phase:
            filtered = filtfilt(b, a, signal)
        else:
            filtered = lfilter(b, a, signal)
    except Exception as e:
        raise RuntimeError(f"Filtre uygulama hatası: {e}")

    return filtered
