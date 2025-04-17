import numpy as np
from typing import Tuple, Union

def histogram_hesapla(image: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Görüntünün histogramını hesaplar.
    RGB görüntü için her kanal için ayrı histogram döndürür.
    Gri görüntü için tek histogram döndürür.
    """
    if len(image.shape) == 3:  # RGB görüntü
        r_hist = np.zeros(256, dtype=int)
        g_hist = np.zeros(256, dtype=int)
        b_hist = np.zeros(256, dtype=int)
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                r_hist[image[i, j, 0]] += 1
                g_hist[image[i, j, 1]] += 1
                b_hist[image[i, j, 2]] += 1
                
        return r_hist, g_hist, b_hist
    else:  # Gri görüntü
        hist = np.zeros(256, dtype=int)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                hist[image[i, j]] += 1
        return hist

def histogram_esitleme(image: np.ndarray, kanal_bazli: bool = False) -> np.ndarray:
    """
    Histogram eşitleme işlemi uygular.
    
    Parametreler:
    image: Giriş görüntüsü
    kanal_bazli: RGB görüntüler için kanalları ayrı ayrı işle
    """
    if len(image.shape) == 3 and kanal_bazli:  # RGB görüntü, kanal bazlı işlem
        result = np.zeros_like(image)
        for i in range(3):  # Her kanal için
            kanal = image[:,:,i]
            hist = histogram_hesapla(kanal)
            
            # Kümülatif histogram hesapla
            cum_hist = np.cumsum(hist)
            
            # Normalizasyon
            cum_hist = ((cum_hist - cum_hist.min()) * 255 / 
                       (cum_hist.max() - cum_hist.min())).astype(np.uint8)
            
            # Görüntüyü dönüştür
            result[:,:,i] = cum_hist[kanal]
            
        return result
    else:  # Gri görüntü veya RGB görüntü (tek kanal olarak işle)
        if len(image.shape) == 3:
            # RGB'yi griye dönüştür
            image = np.mean(image, axis=2).astype(np.uint8)
        
        hist = histogram_hesapla(image)
        cum_hist = np.cumsum(hist)
        cum_hist = ((cum_hist - cum_hist.min()) * 255 / 
                   (cum_hist.max() - cum_hist.min())).astype(np.uint8)
        
        return cum_hist[image]

def histogram_germe(image: np.ndarray, min_deger: int = 0, max_deger: int = 255, 
                   kanal_bazli: bool = False) -> np.ndarray:
    """
    Histogram germe işlemi uygular.
    
    Parametreler:
    image: Giriş görüntüsü
    min_deger: Hedef minimum değer
    max_deger: Hedef maksimum değer
    kanal_bazli: RGB görüntüler için kanalları ayrı ayrı işle
    """
    if len(image.shape) == 3 and kanal_bazli:  # RGB görüntü, kanal bazlı işlem
        result = np.zeros_like(image)
        for i in range(3):  # Her kanal için
            kanal = image[:,:,i]
            kanal_min = np.min(kanal)
            kanal_max = np.max(kanal)
            
            # Germe formülü
            result[:,:,i] = np.clip(
                (kanal - kanal_min) * (max_deger - min_deger) / 
                (kanal_max - kanal_min + 1e-8) + min_deger, 
                0, 255
            ).astype(np.uint8)
            
        return result
    else:  # Gri görüntü veya RGB görüntü (tek kanal olarak işle)
        if len(image.shape) == 3:
            # RGB'yi griye dönüştür
            image = np.mean(image, axis=2).astype(np.uint8)
        
        img_min = np.min(image)
        img_max = np.max(image)
        
        return np.clip(
            (image - img_min) * (max_deger - min_deger) / 
            (img_max - img_min + 1e-8) + min_deger,
            0, 255
        ).astype(np.uint8)

def histogram_belirginlestirme(image: np.ndarray, kanal_bazli: bool = False) -> np.ndarray:
    """
    Histogram belirginleştirme işlemi uygular.
    Kontrast artırma ve detayları belirginleştirme için kullanılır.
    
    Parametreler:
    image: Giriş görüntüsü
    kanal_bazli: RGB görüntüler için kanalları ayrı ayrı işle
    """
    if len(image.shape) == 3 and kanal_bazli:  # RGB görüntü, kanal bazli işlem
        result = np.zeros_like(image)
        for i in range(3):  # Her kanal için
            kanal = image[:,:,i]
            
            # Ortalama ve standart sapma hesapla
            mean = np.mean(kanal)
            std = np.std(kanal)
            
            # Belirginleştirme formülü
            result[:,:,i] = np.clip(
                ((kanal - mean) * (2.0 * std) / (std + 1e-8)) + mean,
                0, 255
            ).astype(np.uint8)
            
        return result
    else:  # Gri görüntü veya RGB görüntü (tek kanal olarak işle)
        if len(image.shape) == 3:
            # RGB'yi griye dönüştür
            image = np.mean(image, axis=2).astype(np.uint8)
        
        mean = np.mean(image)
        std = np.std(image)
        
        return np.clip(
            ((image - mean) * (2.0 * std) / (std + 1e-8)) + mean,
            0, 255
        ).astype(np.uint8)

def histogram_isle(image: np.ndarray, islem_turu: str, 
                  min_deger: int = 0, max_deger: int = 255,
                  kanal_bazli: bool = False) -> np.ndarray:
    """
    Histogram işlemlerini uygular.
    
    Parametreler:
    image: Giriş görüntüsü
    islem_turu: İşlem türü ('esitleme', 'germe', 'belirginlestirme')
    min_deger: Germe işlemi için minimum değer
    max_deger: Germe işlemi için maksimum değer
    kanal_bazli: RGB görüntüler için kanalları ayrı ayrı işle
    
    Dönüş:
    İşlenmiş görüntü
    """
    # İşlemi uygula
    if islem_turu == 'esitleme':
        sonuc = histogram_esitleme(image, kanal_bazli)
    elif islem_turu == 'germe':
        sonuc = histogram_germe(image, min_deger, max_deger, kanal_bazli)
    elif islem_turu == 'belirginlestirme':
        sonuc = histogram_belirginlestirme(image, kanal_bazli)
    else:
        raise ValueError(f"Geçersiz işlem türü: {islem_turu}")
    
    return sonuc

def histogram_grafigi_olustur(hist_verisi: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]], 
                            genislik: int = 800, yukseklik: int = 400) -> np.ndarray:
    """
    Histogram verilerinden görsel bir grafik oluşturur.
    
    Parametreler:
    hist_verisi: Tek kanallı veya RGB histogram verisi
    genislik: Grafik genişliği
    yukseklik: Grafik yüksekliği
    """
    # Kenar boşlukları
    kenar_bosluk = 40
    alt_bosluk = 50
    
    # Çizim alanı boyutları
    cizim_genislik = genislik - 2 * kenar_bosluk
    cizim_yukseklik = yukseklik - kenar_bosluk - alt_bosluk
    
    # Boş bir görüntü oluştur (beyaz arka plan)
    grafik = np.ones((yukseklik, genislik, 3), dtype=np.uint8) * 255
    
    # Eksen çizgileri (siyah)
    grafik[kenar_bosluk:yukseklik-alt_bosluk, kenar_bosluk] = [0, 0, 0]  # Y ekseni
    grafik[yukseklik-alt_bosluk, kenar_bosluk:genislik-kenar_bosluk] = [0, 0, 0]  # X ekseni
    
    if isinstance(hist_verisi, tuple):  # RGB histogram
        renkler = [(255,0,0), (0,255,0), (0,0,255)]  # RGB renkleri
        etiketler = ['R', 'G', 'B']
        
        for idx, (hist, renk, etiket) in enumerate(zip(hist_verisi, renkler, etiketler)):
            # Histogram değerlerini normalize et
            hist = hist * (cizim_yukseklik - 20) / (np.max(hist) + 1e-8)
            
            # Her bin için kalın sütun çiz
            bin_genislik = cizim_genislik / 256
            for i in range(256):
                x1 = int(kenar_bosluk + i * bin_genislik)
                x2 = int(kenar_bosluk + (i + 1) * bin_genislik)
                y1 = int(yukseklik - alt_bosluk)
                y2 = int(yukseklik - alt_bosluk - hist[i])
                
                # Kalın sütun çiz
                grafik[y2:y1, max(x1-1, kenar_bosluk):min(x2+1, genislik-kenar_bosluk)] = [int(c * 0.4) for c in renk]
                
                # Önemli değerleri göster (yüksek frekanslı noktalar)
                if hist[i] > cizim_yukseklik * 0.3:  # Eşik değerini düşürdük
                    deger_str = f"{i}:{hist_verisi[idx][i]}"  # Piksel değeri ve frekansı
                    x_text = x1 - 15
                    y_text = y2 - 15
                    # Kalın yazı için aynı metni birkaç piksel kaydırarak yaz
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            for c in deger_str:
                                grafik[y_text+dy:y_text+dy+8, x_text+dx:x_text+dx+6] = renk
                                x_text += 7
                            x_text = x1 - 15  # Reset x position for next iteration
            
            # Kanal etiketini ekle
            x_etiket = genislik - kenar_bosluk + 10
            y_etiket = kenar_bosluk + idx * 20
            # Kalın yazı için
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    grafik[y_etiket+dy:y_etiket+dy+15, x_etiket+dx:x_etiket+dx+15] = renk
            
    else:  # Tek kanallı histogram
        # Histogram değerlerini normalize et
        hist = hist_verisi * (cizim_yukseklik - 20) / (np.max(hist_verisi) + 1e-8)
        
        # Her bin için kalın sütun çiz
        bin_genislik = cizim_genislik / 256
        for i in range(256):
            x1 = int(kenar_bosluk + i * bin_genislik)
            x2 = int(kenar_bosluk + (i + 1) * bin_genislik)
            y1 = int(yukseklik - alt_bosluk)
            y2 = int(yukseklik - alt_bosluk - hist[i])
            
            # Kalın sütun çiz
            grafik[y2:y1, max(x1-1, kenar_bosluk):min(x2+1, genislik-kenar_bosluk)] = [100, 100, 100]
            
            # Önemli değerleri göster (yüksek frekanslı noktalar)
            if hist[i] > cizim_yukseklik * 0.3:  # Eşik değerini düşürdük
                deger_str = f"{i}:{hist_verisi[i]}"  # Piksel değeri ve frekansı
                x_text = x1 - 15
                y_text = y2 - 15
                # Kalın yazı için aynı metni birkaç piksel kaydırarak yaz
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for c in deger_str:
                            grafik[y_text+dy:y_text+dy+8, x_text+dx:x_text+dx+6] = [0, 0, 0]
                            x_text += 7
                        x_text = x1 - 15  # Reset x position for next iteration
    
    # X ekseni etiketleri
    for i in range(9):
        x_deger = int(255 * i / 8)
        x_pos = int(kenar_bosluk + (cizim_genislik * i / 8))
        grafik[yukseklik-alt_bosluk:yukseklik-alt_bosluk+5, x_pos] = [0, 0, 0]
        deger_str = str(x_deger)
        x_text = x_pos - 10
        y_text = yukseklik - alt_bosluk + 20
        # Kalın yazı için
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for c in deger_str:
                    grafik[y_text+dy:y_text+dy+8, x_text+dx:x_text+dx+6] = [0, 0, 0]
                    x_text += 7
                x_text = x_pos - 10  # Reset x position for next iteration
    
    return grafik 