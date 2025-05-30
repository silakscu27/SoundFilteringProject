import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import stft

def plot_waveform(signal, sample_rate, title="Waveform", save_path=None):
    """
    Plots the waveform in the time domain.
    """
    time = np.arange(len(signal)) / sample_rate
    plt.figure(figsize=(10, 4))
    plt.plot(time, signal)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_fft(signal, sample_rate, title="Frequency Spectrum (FFT)", save_path=None):
    """
    Plots the frequency spectrum using FFT.
    """
    N = len(signal)
    freqs = fftfreq(N, 1 / sample_rate)
    magnitude = np.abs(fft(signal))

    idx = np.where(freqs >= 0)
    freqs = freqs[idx]
    magnitude = magnitude[idx]

    plt.figure(figsize=(10, 4))
    plt.plot(freqs, magnitude)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_comparison(clean, noisy, filtered, sample_rate, save_path=None):
    """
    Plots time-domain comparison of clean, noisy, and filtered signals.
    """
    time = np.arange(len(clean)) / sample_rate
    plt.figure(figsize=(12, 6))

    plt.plot(time, clean, label="Clean", alpha=0.8)
    plt.plot(time, noisy, label="Noisy", alpha=0.6)
    plt.plot(time, filtered, label="Filtered", alpha=0.7)

    plt.title("Time-Domain Signal Comparison")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_spectrogram(signal, sample_rate, n_fft=1024, hop_length=None, title="Spectrogram", save_path=None):
    """
    Plots a spectrogram using STFT.
    """
    if hop_length is None:
        hop_length = n_fft // 4  # 75% overlap

    f, t, Zxx = stft(signal, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop_length)
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
