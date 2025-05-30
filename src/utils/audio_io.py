import os
import soundfile as sf
import numpy as np

def load_audio(file_path):
    """
    Verilen dosya yolundan ses dosyasını yükler.
    Geriye: sinyal (numpy array), örnekleme frekansı
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Ses dosyası bulunamadı: {file_path}")
    
    data, sample_rate = sf.read(file_path)
    return data, sample_rate

def save_audio(file_path, signal, sample_rate):
    """
    İşlenmiş sesi belirlenen yola kaydeder (.wav).
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    sf.write(file_path, signal, sample_rate)

def match_file_pairs(original_dir, noisy_dir):
    """
    Orijinal ve gürültülü klasörler arasında dosya adı eşleşmesi yapar.
    Geriye: (original_path, noisy_path) çiftlerinden oluşan liste
    """
    pairs = []
    noisy_files = os.listdir(noisy_dir)

    for file_name in noisy_files:
        original_path = os.path.join(original_dir, file_name)
        noisy_path = os.path.join(noisy_dir, file_name)
        if os.path.exists(original_path):
            pairs.append((original_path, noisy_path))
    
    return pairs
