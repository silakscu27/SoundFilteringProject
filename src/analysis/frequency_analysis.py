import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import stft

def compute_fft(signal, sample_rate):
    """
    Computes FFT of the signal.
    Returns:
        freqs: Frequency bins (Hz)
        magnitude: Magnitude of the FFT
    """
    N = len(signal)
    freqs = fftfreq(N, d=1/sample_rate)
    fft_values = fft(signal)
    magnitude = np.abs(fft_values)

    # Only positive frequencies
    idx = np.where(freqs >= 0)
    return freqs[idx], magnitude[idx]

def compute_stft(signal, sample_rate, n_fft=1024, hop_length=None):
    """
    Computes Short-Time Fourier Transform (STFT) of the signal.
    Returns:
        f: frequency bins (Hz)
        t: time bins (s)
        Zxx: complex STFT matrix
    """
    if hop_length is None:
        hop_length = n_fft // 4  # default 75% overlap

    f, t, Zxx = stft(signal, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop_length)
    return f, t, Zxx

def plot_spectrogram(signal, sample_rate, n_fft=1024, hop_length=None, title="Spectrogram", save_path=None):
    """
    Plots a spectrogram using STFT.
    """
    f, t, Zxx = compute_stft(signal, sample_rate, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(Zxx)

    plt.figure(figsize=(10, 5))
    plt.pcolormesh(t, f, 20 * np.log10(magnitude + 1e-10), shading='gouraud')
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Magnitude (dB)")

    if save_path:
        plt.savefig(save_path)
    plt.close()
