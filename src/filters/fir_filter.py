import numpy as np
from scipy.signal import firwin, lfilter

def design_fir_filter(filter_type, cutoff, fs, numtaps=101, band=None):
    """
    FIR filtre tasarımı.
    
    Parametreler:
        filter_type: 'lowpass', 'highpass', 'bandpass', 'bandstop'
        cutoff: Tek değerli cutoff (low/high) için [Hz]
        fs: Örnekleme frekansı [Hz]
        numtaps: Filtre derecesi (daha yüksek = daha keskin geçiş)
        band: Bandpass veya bandstop için [low, high] aralığı
    
    Geriye:
        taps: FIR filtresinin ağırlıkları
    """
    nyq = fs / 2
    
    if filter_type == 'lowpass':
        return firwin(numtaps, cutoff / nyq, pass_zero=True)
    elif filter_type == 'highpass':
        return firwin(numtaps, cutoff / nyq, pass_zero=False)
    elif filter_type == 'bandpass':
        if band is None or len(band) != 2:
            raise ValueError("Bandpass requires a band=[low, high] argument.")
        return firwin(numtaps, [band[0] / nyq, band[1] / nyq], pass_zero=False)
    elif filter_type == 'bandstop':
        if band is None or len(band) != 2:
            raise ValueError("Bandstop requires a band=[low, high] argument.")
        return firwin(numtaps, [band[0] / nyq, band[1] / nyq], pass_zero=True)
    else:
        raise ValueError(f"Unsupported filter type: {filter_type}")

def apply_fir_filter(signal, taps):
    """
    FIR filtresini sinyale uygular.
    """
    if not isinstance(signal, np.ndarray):
        raise TypeError("Signal must be a numpy array.")
    
    try:
        filtered = lfilter(taps, 1.0, signal)
    except Exception as e:
        raise RuntimeError(f"FIR filter application failed: {e}")
    
    return filtered
