import os
import numpy as np
import soundfile as sf
import librosa

def load_audio(file_path, target_sr=None, mono=True):
    """
    Ses dosyasını yükler ve temel ön işlemeler yapar
    :param file_path: Ses dosyası yolu
    :param target_sr: Hedef örnekleme oranı (None orijinali korur)
    :param mono: Tek kanala dönüştürülsün mü?
    :return: (audio_data, sample_rate) tuple'ı
    """
    try:
        # Librosa ile dosyayı yükle (dtype=np.float32)
        audio, sr = librosa.load(file_path, sr=target_sr, mono=mono)
        
        # Normalizasyon (peak amplitude 1.0 olacak şekilde)
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
            
        return audio, sr
        
    except Exception as e:
        raise RuntimeError(f"Error loading audio file {file_path}: {str(e)}")

def save_audio(file_path, audio_data, sample_rate):
    """
    Ses dosyasını kaydeder
    :param file_path: Kaydedilecek dosya yolu
    :param audio_data: Ses verisi (numpy array)
    :param sample_rate: Örnekleme oranı
    """
    try:
        # Dizin yoksa oluştur
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # PCM formatında kaydet (32-bit float)
        sf.write(file_path, audio_data, sample_rate, subtype='FLOAT')
        
    except Exception as e:
        raise RuntimeError(f"Error saving audio file {file_path}: {str(e)}")

def resample_audio(audio_data, original_sr, target_sr):
    """
    Ses verisini yeniden örnekler
    :param audio_data: Orijinal ses verisi
    :param original_sr: Orijinal örnekleme oranı
    :param target_sr: Hedef örnekleme oranı
    :return: Yeniden örneklenmiş ses verisi
    """
    if original_sr == target_sr:
        return audio_data
        
    duration = len(audio_data) / original_sr
    new_length = int(duration * target_sr)
    return librosa.resample(audio_data, orig_sr=original_sr, target_sr=target_sr)

def normalize_audio(audio_data, target_level=-3.0):
    """
    Ses verisini normalleştirir (belirli bir dB seviyesine)
    :param audio_data: Ses verisi
    :param target_level: Hedef dB seviyesi (negatif değer)
    :return: Normalleştirilmiş ses verisi
    """
    if len(audio_data) == 0:
        return audio_data
        
    # RMS değerini hesapla
    rms = np.sqrt(np.mean(audio_data**2))
    if rms < 1e-6:  # Sessiz dosya
        return audio_data
        
    # Hedef genliği hesapla (dBFS cinsinden)
    target_amplitude = 10 ** (target_level / 20)
    current_amplitude = np.max(np.abs(audio_data))
    
    # Normalizasyon faktörü
    scaling_factor = target_amplitude / current_amplitude
    return np.clip(audio_data * scaling_factor, -1.0, 1.0)

def stereo_to_mono(audio_data):
    """
    Stereo sesi mono'ya dönüştürür
    :param audio_data: Stereo ses verisi (2D array)
    :return: Mono ses verisi
    """
    if len(audio_data.shape) == 1:
        return audio_data
    return np.mean(audio_data, axis=1)

if __name__ == "__main__":
    # Test kodu
    TEST_MODE = False
    
    if TEST_MODE:
        # Örnek kullanım
        test_file = os.path.join('data', 'original', 'test.wav')
        
        # Yükleme testi
        audio, sr = load_audio(test_file)
        print(f"Loaded audio: {len(audio)} samples at {sr}Hz")
        
        # Yeniden örnekleme testi
        resampled = resample_audio(audio, sr, sr//2)
        print(f"Resampled to: {len(resampled)} samples at {sr//2}Hz")
        
        # Kaydetme testi
        save_path = os.path.join('results', 'test_output.wav')
        save_audio(save_path, resampled, sr//2)
        print(f"Audio saved to: {save_path}")