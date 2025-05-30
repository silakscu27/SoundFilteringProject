import os
from utils.audio_io import load_audio, save_audio, match_file_pairs
from filters.iir_filter import design_iir_filter, apply_iir_filter
from utils.metrics import mean_squared_error, signal_to_noise_ratio, peak_signal_to_noise_ratio, correlation_coefficient
from utils.visualization import plot_waveform, plot_fft, plot_comparison, plot_spectrogram
from analysis.time_analysis import compute_duration, compute_rms, compute_zero_crossing_rate

# Konfigürasyon
ORIGINAL_DIR = "data/original"
NOISY_DIR = "data/noisy"
OUTPUT_DIR = "results/filtered_audio"
PLOT_DIR = "results/plots"

FILTER_BAND = [400, 1500]  # Gürültü frekansları (örnek)
FILTER_ORDER = 4

def process_file(original_path, noisy_path, output_dir):
    clean, sr = load_audio(original_path)
    noisy, _ = load_audio(noisy_path)

    # Filtre tasarımı ve uygulaması (IIR band-stop)
    b, a = design_iir_filter('bandstop', cutoff=None, fs=sr, order=FILTER_ORDER, band=FILTER_BAND)
    filtered = apply_iir_filter(noisy, b, a)

    # Kaydet
    base_filename = os.path.basename(noisy_path)
    filtered_path = os.path.join(output_dir, base_filename)
    save_audio(filtered_path, filtered, sr)

    # Görseller
    plot_waveform(noisy, sr, title="Noisy Waveform", save_path=f"{PLOT_DIR}/{base_filename}_noisy_wave.png")
    plot_fft(noisy, sr, title="Noisy FFT", save_path=f"{PLOT_DIR}/{base_filename}_noisy_fft.png")
    plot_spectrogram(noisy, sr, title="Noisy Spectrogram", save_path=f"{PLOT_DIR}/{base_filename}_noisy_spec.png")
    plot_comparison(clean, noisy, filtered, sr, save_path=f"{PLOT_DIR}/{base_filename}_comparison.png")

    # Metrikler
    mse = mean_squared_error(clean, filtered)
    snr = signal_to_noise_ratio(clean, filtered)
    psnr = peak_signal_to_noise_ratio(clean, filtered)
    corr = correlation_coefficient(clean, filtered)

    # Zaman analizleri (rapor için kullanılabilir)
    duration = compute_duration(clean, sr)
    rms_before = compute_rms(noisy)
    rms_after = compute_rms(filtered)
    zcr_before = compute_zero_crossing_rate(noisy)
    zcr_after = compute_zero_crossing_rate(filtered)

    # Sonuçları döndür
    return {
        "file": base_filename,
        "mse": mse,
        "snr": snr,
        "psnr": psnr,
        "corr": corr,
        "duration": duration,
        "rms_before": rms_before,
        "rms_after": rms_after,
        "zcr_before": zcr_before,
        "zcr_after": zcr_after,
    }

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    pairs = match_file_pairs(ORIGINAL_DIR, NOISY_DIR)
    results = []

    print(f"Processing {len(pairs)} file(s)...")

    for original_path, noisy_path in pairs:
        try:
            result = process_file(original_path, noisy_path, OUTPUT_DIR)
            results.append(result)
            print(f"✔ Processed {result['file']} | SNR: {result['snr']:.2f} dB, Corr: {result['corr']:.3f}")
        except Exception as e:
            print(f"✖ Error processing {noisy_path}: {e}")

    print("\nSummary:")
    for r in results:
        print(f"{r['file']}: MSE={r['mse']:.4f}, SNR={r['snr']:.2f}, Corr={r['corr']:.3f}")

if __name__ == "__main__":
    main()
