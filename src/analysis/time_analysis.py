import numpy as np

def compute_duration(signal, sample_rate):
    """
    Computes the duration (in seconds) of the signal.
    """
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")
    return len(signal) / sample_rate

def compute_rms(signal):
    """
    Computes Root Mean Square (RMS) value of the signal.
    Indicates signal power.
    """
    return np.sqrt(np.mean(signal ** 2))

def compute_zero_crossing_rate(signal):
    """
    Computes the zero-crossing rate of the signal.
    Indicates frequency of sign changes (e.g., noisy vs steady).
    """
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    return len(zero_crossings) / len(signal)

def compute_amplitude_range(signal):
    """
    Computes peak-to-peak amplitude range of the signal.
    """
    return np.max(signal) - np.min(signal)
