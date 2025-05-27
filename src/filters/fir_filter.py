import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os
from utils.audio_io import load_audio, save_audio
from utils.visualization import plot_frequency_response

class FIRFilter:
    """
    Özelleştirilebilir FIR filtre tasarımı ve uygulaması
    """
    
    def __init__(self, sample_rate, cutoff_freq, filter_type='lowpass', order=101, window='hamming'):
        """
        FIR filtresi tasarlar
        
        :param sample_rate: Örnekleme frekansı (Hz)
        :param cutoff_freq: Kesim frekansı (tek değer veya [low, high] listesi)
        :param filter_type: 'lowpass', 'highpass', 'bandpass', 'bandstop'
        :param order: Filtre derecesi (katsayı sayısı)
        :param window: Pencereleme fonksiyonu ('hamming', 'hann', 'blackman', 'kaiser')
        """
        self.sample_rate = sample_rate
        self.cutoff_freq = cutoff_freq
        self.filter_type = filter_type
        self.order = order
        self.window = window
        self.coefficients = self._design_filter()
        
    def _design_filter(self):
        """
        FIR filtre katsayılarını tasarlar
        """
        nyquist = 0.5 * self.sample_rate
        
        # Kesim frekanslarını normalize et
        if isinstance(self.cutoff_freq, (list, tuple, np.ndarray)):
            normalized_cutoff = [f / nyquist for f in self.cutoff_freq]
        else:
            normalized_cutoff = self.cutoff_freq / nyquist
        
        # Filtre tipine göre tasarım yap
        if self.filter_type == 'lowpass':
            coefficients = signal.firwin(self.order, 
                                       normalized_cutoff, 
                                       window=self.window,
                                       fs=self.sample_rate)
        elif self.filter_type == 'highpass':
            coefficients = signal.firwin(self.order, 
                                       normalized_cutoff, 
                                       pass_zero=False,
                                       window=self.window,
                                       fs=self.sample_rate)
        elif self.filter_type == 'bandpass':
            coefficients = signal.firwin(self.order, 
                                       normalized_cutoff, 
                                       pass_zero=False,
                                       window=self.window,
                                       fs=self.sample_rate)
        elif self.filter_type == 'bandstop':
            coefficients = signal.firwin(self.order, 
                                       normalized_cutoff, 
                                       pass_zero=False,
                                       window=self.window,
                                       fs=self.sample_rate)
        else:
            raise ValueError("Geçersiz filtre tipi. 'lowpass', 'highpass', 'bandpass' veya 'bandstop' olmalı")
        
        return coefficients
    
    def apply(self, audio_data):
        """
        Filtreyi ses sinyaline uygular
        
        :param audio_data: Giriş ses sinyali
        :return: Filtrelenmiş ses sinyali
        """
        filtered = signal.lfilter(self.coefficients, 1.0, audio_data)
        
        # Faz gecikmesini düzelt (sabit gecikme ekleyerek)
        filtered = np.roll(filtered, -self.order // 2)
        
        return filtered
    
    def plot_response(self, save_path=None):
        """
        Filtrenin frekans tepkisini çizer
        
        :param save_path: Kaydedilecek dosya yolu (None ise göstermez)
        """
        w, h = signal.freqz(self.coefficients, fs=self.sample_rate)
        
        plt.figure(figsize=(10, 6))
        
        # Genlik tepkisi (dB)
        plt.subplot(2, 1, 1)
        plt.plot(w, 20 * np.log10(np.abs(h) + 1e-10))
        plt.title(f'FIR {self.filter_type} Filter Frequency Response\nCutoff: {self.cutoff_freq}Hz, Order: {self.order}')
        plt.ylabel('Amplitude (dB)')
        plt.grid(True)
        
        # Kesim frekanslarını işaretle
        if isinstance(self.cutoff_freq, (list, tuple, np.ndarray)):
            for fc in self.cutoff_freq:
                plt.axvline(fc, color='red', linestyle='--', alpha=0.5)
        else:
            plt.axvline(self.cutoff_freq, color='red', linestyle='--', alpha=0.5)
        
        # Faz tepkisi
        plt.subplot(2, 1, 2)
        plt.plot(w, np.unwrap(np.angle(h)))
        plt.ylabel('Phase (radians)')
        plt.xlabel('Frequency (Hz)')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()

def design_and_apply_fir(input_path, output_dir, cutoff, filter_type, order=101, window='hamming'):
    """
    FIR filtresi tasarlar, uygular ve sonuçları kaydeder
    
    :param input_path: Giriş ses dosyası yolu
    :param output_dir: Çıktı dizini
    :param cutoff: Kesim frekansı (Hz)
    :param filter_type: Filtre tipi
    :param order: Filtre derecesi
    :param window: Pencereleme fonksiyonu
    :return: Filtrelenmiş ses verisi
    """
    # Ses dosyasını yükle
    audio, sr = load_audio(input_path)
    
    # Filtreyi tasarla
    fir_filter = FIRFilter(sr, cutoff, filter_type, order, window)
    
    # Frekans tepkisini kaydet
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f'fir_{filter_type}_response.png')
    fir_filter.plot_response(save_path=plot_path)
    
    # Filtreyi uygula
    filtered = fir_filter.apply(audio)
    
    # Sonucu kaydet
    output_path = os.path.join(output_dir, f'fir_{filter_type}_filtered.wav')
    save_audio(output_path, filtered, sr)
    
    return filtered
