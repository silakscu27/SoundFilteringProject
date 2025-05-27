import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os
from utils.audio_io import load_audio, save_audio
from utils.visualization import plot_frequency_response

class BandStopFilter:
    """
    Özelleştirilebilir bant durdurma filtresi tasarımı ve uygulaması
    """
    
    def __init__(self, sample_rate, stop_bands, filter_type='fir', order=101):
        """
        Bant durdurma filtresi tasarlar
        
        :param sample_rate: Örnekleme frekansı (Hz)
        :param stop_bands: Durdurulacak frekans bantları [(low1, high1), (low2, high2), ...]
        :param filter_type: 'fir' veya 'iir'
        :param order: Filtre derecesi (FIR için katsayı sayısı)
        """
        self.sample_rate = sample_rate
        self.stop_bands = stop_bands
        self.filter_type = filter_type
        self.order = order
        self.coefficients = self._design_filter()
        
    def _design_filter(self):
        """
        Filtre katsayılarını tasarlar
        """
        nyquist = 0.5 * self.sample_rate
        normalized_bands = []
        
        # Tüm durdurma bantlarını normalize et ve birleştir
        for low, high in self.stop_bands:
            normalized_low = max(0, low / nyquist)
            normalized_high = min(1.0, high / nyquist)
            normalized_bands.extend([normalized_low, normalized_high])
        
        if self.filter_type == 'fir':
            # FIR filtresi tasarla (Kaiser penceresi ile)
            coefficients = signal.firwin(self.order, 
                                       normalized_bands, 
                                       pass_zero=False, 
                                       window='hamming',
                                       fs=self.sample_rate)
        else:
            # IIR filtresi tasarla (Butterworth)
            sos = []
            for i in range(0, len(normalized_bands), 2):
                low = normalized_bands[i]
                high = normalized_bands[i+1]
                if high - low > 0.01:  # Minimum bant genişliği kontrolü
                    sos.append(signal.butter(4, [low, high], 
                                           btype='bandstop', 
                                           output='sos',
                                           fs=self.sample_rate))
            
            if not sos:
                raise ValueError("Geçerli durdurma bandı tanımlanmadı")
            
            # Birden fazla band varsa zincirle
            if len(sos) > 1:
                coefficients = sos[0]
                for s in sos[1:]:
                    coefficients = signal.sosfilt_zi(s)  # Durum bilgisini koru
            else:
                coefficients = sos[0]
        
        return coefficients
    
    def apply(self, audio_data):
        """
        Filtreyi ses sinyaline uygular
        
        :param audio_data: Giriş ses sinyali
        :return: Filtrelenmiş ses sinyali
        """
        if self.filter_type == 'fir':
            filtered = signal.lfilter(self.coefficients, 1.0, audio_data)
        else:
            filtered = signal.sosfilt(self.coefficients, audio_data)
        
        # Faz gecikmesini düzelt (FIR için)
        if self.filter_type == 'fir':
            filtered = np.roll(filtered, -self.order // 2)
        
        return filtered
    
    def plot_response(self, save_path=None):
        """
        Filtrenin frekans tepkisini çizer
        
        :param save_path: Kaydedilecek dosya yolu (None ise göstermez)
        """
        if self.filter_type == 'fir':
            w, h = signal.freqz(self.coefficients, fs=self.sample_rate)
        else:
            w, h = signal.sosfreqz(self.coefficients, fs=self.sample_rate)
        
        plt.figure(figsize=(10, 5))
        
        # Genlik tepkisi (dB)
        plt.subplot(2, 1, 1)
        plt.plot(w, 20 * np.log10(np.abs(h) + 1e-10))
        plt.title('Band-Stop Filter Frequency Response')
        plt.ylabel('Amplitude (dB)')
        plt.grid(True)
        
        # Durdurma bantlarını işaretle
        for low, high in self.stop_bands:
            plt.axvspan(low, high, color='red', alpha=0.1)
        
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

def design_and_test_filter(audio_path, stop_bands, output_dir='results/filtered'):
    """
    Filtre tasarlar, test eder ve sonuçları kaydeder
    
    :param audio_path: İşlenecek ses dosyası yolu
    :param stop_bands: Durdurulacak frekans bantları
    :param output_dir: Çıktı dizini
    """
    # Ses dosyasını yükle
    audio, sr = load_audio(audio_path)
    
    # Filtreyi tasarla
    bsf = BandStopFilter(sr, stop_bands, filter_type='fir', order=201)
    
    # Frekans tepkisini kaydet
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'filter_response.png')
    bsf.plot_response(save_path=plot_path)
    
    # Filtreyi uygula
    filtered = bsf.apply(audio)
    
    # Sonucu kaydet
    output_path = os.path.join(output_dir, 'filtered_audio.wav')
    save_audio(output_path, filtered, sr)
    
    return filtered

if __name__ == "__main__":
    # Test kodu
    TEST_MODE = False
    
    if TEST_MODE:
        # Test parametreleri
        test_audio = os.path.join('data', 'noisy', 'test.wav')
        stop_freqs = [(1000, 3000)]  # 1kHz-3kHz arasını durdur
        
        # Filtreleme işlemini gerçekleştir
        filtered_audio = design_and_test_filter(test_audio, stop_freqs)
        
        print("Filtreleme işlemi tamamlandı. Sonuçlar 'results/filtered' klasörüne kaydedildi.")