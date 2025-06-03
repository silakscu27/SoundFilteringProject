import os
from .analysis.frequency_survey import run_frequency_survey
from .filters.iir_filter import design_iir_filter, apply_iir_filter
from .utils.audio_io import load_audio, save_audio, match_file_pairs
from .utils.metrics import mean_squared_error, signal_to_noise_ratio, correlation_coefficient
from .utils.visualization import plot_comparison, plot_waveform, plot_fft, plot_spectrogram

# Ayarlar
NOISY_DIR = "data/noisy"
ORIG_DIR = "data/original"
FILTERED_DIR = "results/filtered_audio"
PLOT_DIR = "results/plots"
SUMMARY_FILE = "results/summary.txt"
LOW_CUT = 6000
HIGH_CUT = 80
ORDER = 4

def batch_filter_and_evaluate():
    os.makedirs(FILTERED_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    summary_lines = []

    file_pairs = match_file_pairs(ORIG_DIR, NOISY_DIR)

    for orig_path, noisy_path in file_pairs:
        fname = os.path.basename(noisy_path)
        signal, sr = load_audio(noisy_path)
        clean, _ = load_audio(orig_path)

        # Filtreleme
        b_hp, a_hp = design_iir_filter("highpass", fs=sr, cutoff=HIGH_CUT, order=ORDER)
        b_lp, a_lp = design_iir_filter("lowpass", fs=sr, cutoff=LOW_CUT, order=ORDER)

        filtered = apply_iir_filter(signal, b_hp, a_hp, zero_phase=True)
        filtered = apply_iir_filter(filtered, b_lp, a_lp, zero_phase=True)

        # Kaydet
        out_path = os.path.join(FILTERED_DIR, fname.replace(".wav", "_filtered.wav"))
        save_audio(out_path, filtered, sr)

        # Metrikler
        mse = mean_squared_error(clean, filtered)
        snr = signal_to_noise_ratio(clean, filtered)
        corr = correlation_coefficient(clean, filtered)

        summary_lines.append(f"{fname}: MSE={mse:.4f}, SNR={snr:.2f}, Corr={corr:.3f}")

        # Opsiyonel: görselleştirme örneği sadece ilk 1 dosya için
        if fname.startswith("0") or fname.startswith("01"):
            plot_comparison(clean, signal, filtered, sr,
                            save_path=os.path.join(PLOT_DIR, f"{fname}_comparison.png"))
            plot_spectrogram(signal, sr, title="Noisy Spectrogram",
                             save_path=os.path.join(PLOT_DIR, f"{fname}_noisy_spec.png"))
            plot_spectrogram(filtered, sr, title="Filtered Spectrogram",
                             save_path=os.path.join(PLOT_DIR, f"{fname}_filtered_spec.png"))

    with open(SUMMARY_FILE, "w") as f:
        for line in summary_lines:
            f.write(line + "\n")

    print("✔️ Filtreleme ve metrik hesaplama tamamlandı.")
    print(f"📄 Metrikler: {SUMMARY_FILE}")

def main():
    print("🎯 [1] Frekans Analizi Başlatılıyor...")
    run_frequency_survey()

    print("\n🎛️ [2] Filtreleme ve Değerlendirme Başlatılıyor...")
    batch_filter_and_evaluate()

    print("\n✅ Tüm işlemler başarıyla tamamlandı.")
    print("📂 Sonuçlar: results/ klasörü altında.")

if __name__ == "__main__":
    main()
