import numpy as np
from scipy.signal import butter, lfilter

def design_iir_filter(filter_type, cutoff, fs, order=4, band=None):
    """
    IIR filtre tasarımı (Butterworth).
    
    Parametreler:
        filter_type: 'lowpass', 'highpass', 'bandpass', 'bandstop'
        cutoff: low/high filtrelerde sınır frekansı [Hz]
        fs: örnekleme frekansı [Hz]
        order: filtre derecesi
        band: bandpass/bandstop için [low, high] frekansları
    
    Geriye:
        b, a: Filtre katsayıları
    """
    nyq = fs / 2

    if filter_type == 'lowpass':
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
    elif filter_type == 'highpass':
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
    elif filter_type == 'bandpass':
        if band is None or len(band) != 2:
            raise ValueError("Bandpass requires a band=[low, high] argument.")
        normal_band = [band[0] / nyq, band[1] / nyq]
        b, a = butter(order, normal_band, btype='band', analog=False)
    elif filter_type == 'bandstop':
        if band is None or len(band) != 2:
            raise ValueError("Bandstop requires a band=[low, high] argument.")
        normal_band = [band[0] / nyq, band[1] / nyq]
        b, a = butter(order, normal_band, btype='bandstop', analog=False)
    else:
        raise ValueError(f"Unsupported filter type: {filter_type}")

    return b, a

def apply_iir_filter(signal, b, a):
    """
    IIR filtresini sinyale uygular.
    """
    if not isinstance(signal, np.ndarray):
        raise TypeError("Signal must be a numpy array.")
    
    try:
        filtered = lfilter(b, a, signal)
    except Exception as e:
        raise RuntimeError(f"IIR filter application failed: {e}")

    return filtered
