import numpy as np

def mean_squared_error(clean, processed):
    """
    Ortalama kare hatası (MSE)
    """
    if clean.shape != processed.shape:
        raise ValueError("Temiz ve işlenmiş sinyalin boyutları eşleşmiyor.")
    
    return np.mean((clean - processed) ** 2)

def signal_to_noise_ratio(clean, processed):
    """
    SNR (dB): Sinyal-gürültü oranı
    """
    if clean.shape != processed.shape:
        raise ValueError("Temiz ve işlenmiş sinyalin boyutları eşleşmiyor.")
    
    noise = clean - processed
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    return 10 * np.log10(signal_power / noise_power)

def peak_signal_to_noise_ratio(clean, processed):
    """
    PSNR (dB): Tepe sinyal-gürültü oranı
    """
    mse = mean_squared_error(clean, processed)
    if mse == 0:
        return float('inf')

    peak = np.max(np.abs(clean))
    return 20 * np.log10(peak / np.sqrt(mse))

def correlation_coefficient(clean, processed):
    """
    Pearson korelasyon katsayısı
    """
    if clean.shape != processed.shape:
        raise ValueError("Temiz ve işlenmiş sinyalin boyutları eşleşmiyor.")

    return np.corrcoef(clean, processed)[0, 1]
