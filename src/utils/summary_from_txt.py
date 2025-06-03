import re
import numpy as np
import os

def parse_summary_txt(path):
    """
    Parses summary.txt file and returns list of metric dicts
    """
    results = []
    with open(path, "r") as f:
        for line in f:
            match = re.match(r"(.+?): MSE=([-\d.]+), SNR=([-\d.]+), Corr=([-\d.]+)", line.strip())
            if match:
                filename, mse, snr, corr = match.groups()
                results.append({
                    "file": filename,
                    "mse": float(mse),
                    "snr": float(snr),
                    "corr": float(corr)
                })
    return results

def summarize_parsed_metrics(results):
    """
    Computes overall statistics from parsed result dicts and prints them.
    """
    mse_list = [r["mse"] for r in results]
    snr_list = [r["snr"] for r in results]
    corr_list = [r["corr"] for r in results]

    print(f"📊 Overall Metrics from summary.txt")
    print(f"Average MSE  : {np.mean(mse_list):.4f}")
    print(f"Average SNR  : {np.mean(snr_list):.2f} dB")
    print(f"Average Corr : {np.mean(corr_list):.3f}")

    print(f"\nMSE Std Dev  : {np.std(mse_list):.4f}")
    print(f"SNR Min/Max  : {np.min(snr_list):.2f} / {np.max(snr_list):.2f}")
    print(f"Corr Median  : {np.median(corr_list):.3f}")

if __name__ == "__main__":
    summary_path = "results/summary.txt"
    if os.path.exists(summary_path):
        results = parse_summary_txt(summary_path)
        summarize_parsed_metrics(results)
    else:
        print("❌ summary.txt bulunamadı.")
