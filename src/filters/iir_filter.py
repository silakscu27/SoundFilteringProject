import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os
from utils.audio_io import load_audio, save_audio
from utils.visualization import plot_frequency_response

class IIRFilter:
    """
    Özelleştirilebilir IIR filtre tasarımı ve uygulaması
    
    Desteklenen filtre tipleri:
    - Butterworth
    - Chebyshev Tip I
    - Chebyshev Tip II
    - Eliptik
    - Bessel
    """
    
    def __init__(self, sample_rate, cutoff_freq, filter_type='lowpass', 
                 iir_type='butterworth', order=4, rp=1, rs=40):
        """
        IIR filtresi tasarlar
        
        :param sample_rate: Örnekleme frekansı (Hz)
        :param cutoff_freq: Kesim frekansı (tek değer veya [low, high] listesi)
        :param filter_type: 'lowpass', 'highpass', 'bandpass', 'bandstop'
        :param iir_type: 'butterworth', 'cheby1', 'cheby2', 'ellip', 'bessel'
        :param order: Filtre derecesi
        :param rp: Chebyshev Tip I ve Eliptik için geçiş bandı dalgalanması (dB)
        :param rs: Chebyshev Tip II ve Eliptik için durdurma bandı zayıflaması (dB)
        """
        self.sample_rate = sample_rate
        self.cutoff_freq = cutoff_freq
        self.filter_type = filter_type
        self.iir_type = iir_type.lower()
        self.order = order
        self.rp = rp
        self.rs = rs
        self.sos = self._design_filter()
        
    def _design_filter(self):
        """
        IIR filtre katsayılarını tasarlar (Second-Order Sections formatında)
        """
        nyquist = 0.5 * self.sample_rate
        
        # Kesim frekanslarını normalize et
        if isinstance(self.cutoff_freq, (list, tuple, np.ndarray)):
            normalized_cutoff = [f / nyquist for f in self.cutoff_freq]
        else:
            normalized_cutoff = self.cutoff_freq / nyquist
        
        # Filtre tipine göre tasarım yap
        if self.iir_type == 'butterworth':
            sos = signal.butter(self.order, 
                              normalized_cutoff, 
                              btype=self.filter_type,
                              output='sos',
                              fs=self.sample_rate)
        elif self.iir_type == 'cheby1':
            sos = signal.cheby1(self.order, 
                              self.rp,
                              normalized_cutoff, 
                              btype=self.filter_type,
                              output='sos',
                              fs=self.sample_rate)
        elif self.iir_type == 'cheby2':
            sos = signal.cheby2(self.order, 
                              self.rs,
                              normalized_cutoff, 
                              btype=self.filter_type,
                              output='sos',
                              fs=self.sample_rate)
        elif self.iir_type == 'ellip':
            sos = signal.ellip(self.order, 
                             self.rp,
                             self.rs,
                             normalized_cutoff, 
                             btype=self.filter_type,
                             output='sos',
                             fs=self.sample_rate)
        elif self.iir_type == 'bessel':
            sos = signal.bessel(self.order, 
                              normalized_cutoff, 
                              btype=self.filter_type,
                              output='sos',
                              fs=self.sample_rate)
        else:
            raise ValueError("Geçersiz IIR tipi. 'butterworth', 'cheby1', 'cheby2', 'ellip' veya 'bessel' olmalı")
        
        return sos
    
    def apply(self, audio_data):
        """
        Filtreyi ses sinyaline uygular (sabit fazlı filtreleme)
        
        :param audio_data: Giriş ses sinyali
        :return: Filtrelenmiş ses sinyali
        """
        # İleri-geri filtreleme ile faz bozulmasını önle
        filtered = signal.sosfiltfilt(self.sos, audio_data)
        return filtered
    
    def plot_response(self, save_path=None):
        """
        Filtrenin frekans tepkisini çizer
        
        :param save_path: Kaydedilecek dosya yolu (None ise göstermez)
        """
        w, h = signal.sosfreqz(self.sos, fs=self.sample_rate)
        
        plt.figure(figsize=(10, 6))
        
        # Genlik tepkisi (dB)
        plt.subplot(2, 1, 1)
        plt.plot(w, 20 * np.log10(np.abs(h) + 1e-10))
        plt.title(f'{self.iir_type.capitalize()} {self.filter_type} Filter\nCutoff: {self.cutoff_freq}Hz, Order: {self.order}')
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

def design_and_apply_iir(input_path, output_dir, cutoff, filter_type, 
                        iir_type='butterworth', order=4, rp=1, rs=40):
    """
    IIR filtresi tasarlar, uygular ve sonuçları kaydeder
    
    :param input_path: Giriş ses dosyası yolu
    :param output_dir: Çıktı dizini
    :param cutoff: Kesim frekansı (Hz)
    :param filter_type: Filtre tipi
    :param iir_type: IIR filtre tipi
    :param order: Filtre derecesi
    :param rp: Geçiş bandı dalgalanması (dB)
    :param rs: Durdurma bandı zayıflaması (dB)
    :return: Filtrelenmiş ses verisi
    """
    # Ses dosyasını yükle
    audio, sr = load_audio(input_path)
    
    # Filtreyi tasarla
    iir_filter = IIRFilter(sr, cutoff, filter_type, iir_type, order, rp, rs)
    
    # Frekans tepkisini kaydet
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f'iir_{iir_type}_{filter_type}_response.png')
    iir_filter.plot_response(save_path=plot_path)
    
    # Filtreyi uygula
    filtered = iir_filter.apply(audio)
    
    # Sonucu kaydet
    output_path = os.path.join(output_dir, f'iir_{iir_type}_{filter_type}_filtered.wav')
    save_audio(output_path, filtered, sr)
    
    return filtered

if __name__ == "__main__":
    # Test kodu
    TEST_MODE = False
    
    if TEST_MODE:
        # Test parametreleri
        test_audio = os.path.join('data', 'noisy', 'test.wav')
        output_dir = os.path.join('results', 'iir_filtered')
        
        # 1. Butterworth alçak geçiren filtre testi
        print("Butterworth alçak geçiren filtre uygulanıyor...")
        design_and_apply_iir(test_audio, output_dir, 
                           cutoff=2000, 
                           filter_type='lowpass',
                           iir_type='butterworth',
                           order=4)
        
        # 2. Chebyshev Tip I yüksek geçiren filtre testi
        print("Chebyshev Tip I yüksek geçiren filtre uygulanıyor...")
        design_and_apply_iir(test_audio, output_dir, 
                           cutoff=500, 
                           filter_type='highpass',
                           iir_type='cheby1',
                           order=5,
                           rp=0.5)
        
        # 3. Eliptik bant durduran filtre testi
        print("Eliptik bant durduran filtre uygulanıyor...")
        design_and_apply_iir(test_audio, output_dir, 
                           cutoff=[1000, 3000], 
                           filter_type='bandstop',
                           iir_type='ellip',
                           order=6,
                           rp=1,
                           rs=60)
        
        print("IIR filtre testleri tamamlandı. Sonuçlar 'results/iir_filtered' klasörüne kaydedildi.")