import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

SUMMARY_FILE = "results/summary.txt"
PLOT_DIR = "results/plots/summary_plots"

os.makedirs(PLOT_DIR, exist_ok=True)

def parse_summary_line(line):
    try:
        filename, rest = line.strip().split(":")
        parts = rest.strip().split(",")
        mse = float(parts[0].split("=")[1])
        snr = float(parts[1].split("=")[1])
        corr = float(parts[2].split("=")[1])
        return {"file": filename, "mse": mse, "snr": snr, "corr": corr}
    except:
        return None

def generate_summary_plots():
    with open(SUMMARY_FILE, "r") as f:
        parsed = [parse_summary_line(line) for line in f if parse_summary_line(line)]
    if not parsed:
        print("❌ summary.txt boş ya da bozuk.")
        return

    mse_list = [p["mse"] for p in parsed]
    snr_list = [p["snr"] for p in parsed]
    corr_list = [p["corr"] for p in parsed]

    # Bar plot (average metrics)
    plt.figure(figsize=(8, 4))
    plt.bar(["MSE", "SNR (dB)", "Corr"], [np.mean(mse_list), np.mean(snr_list), np.mean(corr_list)],
            color=["skyblue", "salmon", "limegreen"])
    plt.title("Average Denoising Metrics")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/overall_metrics_barplot.png")
    plt.close()

    # SNR histogram
    plt.figure(figsize=(8, 4))
    plt.hist(snr_list, bins=30, color="salmon", edgecolor="black")
    plt.title("SNR Distribution")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Number of Files")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/snr_distribution.png")
    plt.close()

    # Correlation KDE
    plt.figure(figsize=(8, 4))
    sns.kdeplot(data=corr_list, fill=True, label="Correlation")
    plt.title("Correlation Density")
    plt.xlabel("Correlation")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/correlation_density.png")
    plt.close()

    print("📊 Grafikler oluşturuldu:", PLOT_DIR)

if __name__ == "__main__":
    generate_summary_plots()
