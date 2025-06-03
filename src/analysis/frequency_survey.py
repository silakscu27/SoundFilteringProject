import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from ..utils.audio_io import load_audio

def compute_fft(signal, sample_rate):
    N = len(signal)
    freqs = fftfreq(N, 1 / sample_rate)
    magnitude = np.abs(fft(signal))
    idx = np.where(freqs >= 0)
    return freqs[idx], magnitude[idx]

def run_frequency_survey(folder_path="data/noisy", save_plot_path="results/plots/mean_fft.png", summary_csv="results/reports/frequency_summary.csv"):
    all_magnitudes = []
    reference_freqs = None

    for filename in os.listdir(folder_path):
        if not filename.endswith(".wav"):
            continue

        path = os.path.join(folder_path, filename)
        signal, sr = load_audio(path)

        freqs, mag = compute_fft(signal, sr)

        if reference_freqs is None:
            reference_freqs = freqs
        else:
            if not np.allclose(freqs, reference_freqs):
                print(f"Skipping {filename}, incompatible frequency bins.")
                continue

        all_magnitudes.append(mag)

    all_magnitudes = np.array(all_magnitudes)
    avg_magnitude = np.mean(all_magnitudes, axis=0)


    plt.figure(figsize=(12, 5))
    plt.plot(reference_freqs, avg_magnitude)
    plt.title("Average Frequency Spectrum for All Noisy Files")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Average Amplitude")
    plt.grid(True)
    plt.savefig(save_plot_path)
    plt.close()

    # Numerical analysis: frequencies with intense energy
    threshold = np.max(avg_magnitude) * 0.1
    dominant_freqs = reference_freqs[avg_magnitude > threshold]

    # Write numeric output as CSV
    import pandas as pd
    df = pd.DataFrame({
        "frequency": reference_freqs,
        "mean_magnitude": avg_magnitude
    })
    df.to_csv(summary_csv, index=False)

    print("The average frequency analysis is completed.")
    print(f"Dominant frequency ranges (Amplitude > 10% max):\n{dominant_freqs}")
