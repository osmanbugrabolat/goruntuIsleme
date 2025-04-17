import numpy as np
from .gray_scale import rgb_to_gray

def adaptif_esikleme(gri_goruntu: np.ndarray, blok_boyutu: int = 7) -> np.ndarray:
    """
    Adaptif eşikleme yöntemi uygular.
    Her piksel için çevresindeki piksellerin ortalamasını eşik değeri olarak kullanır.
    Saf NumPy kullanarak implementasyon.
    """
    if blok_boyutu % 2 == 0:
        blok_boyutu += 1  # Blok boyutu tek sayı olmalı
        
    padding = blok_boyutu // 2
    height, width = gri_goruntu.shape
    
    # Görüntüyü padding ile genişlet
    padded = np.pad(gri_goruntu, padding, mode='edge')
    
    # Çıktı görüntüsü için boş dizi oluştur
    binary = np.zeros_like(gri_goruntu)
    
    # Her piksel için lokal ortalamayı hesapla
    for i in range(height):
        for j in range(width):
            # Pencere sınırlarını belirle
            y_start = i
            y_end = i + blok_boyutu
            x_start = j
            x_end = j + blok_boyutu
            
            # Penceredeki değerlerin ortalamasını hesapla
            pencere = padded[y_start:y_end, x_start:x_end]
            ortalama = np.mean(pencere)
            
            # Eşikleme uygula (offset değeri ile)
            binary[i, j] = 255 if gri_goruntu[i, j] > ortalama - 10 else 0
    
    return binary

def binary_donusum(image: np.ndarray, threshold: int = 127, method: str = 'basit', blok_boyutu: int = 7) -> np.ndarray:
    """
    Görüntüyü binary formata dönüştürür.
    Hiçbir hazır fonksiyon kullanılmadan implementasyon.
    
    Args:
        image (np.ndarray): Giriş görüntüsü (RGB veya Gri)
        threshold (int): Eşik değeri (0-255 arası), sadece basit eşiklemede kullanılır
        method (str): Eşikleme yöntemi ('basit', 'otsu', 'adaptif')
        blok_boyutu (int): Adaptif eşikleme için blok boyutu
    
    Returns:
        np.ndarray: Binary formata dönüştürülmüş görüntü
    """
    # Görüntüyü gri tonlamaya çevir
    if len(image.shape) == 3:
        gri_goruntu = rgb_to_gray(image)
    else:
        gri_goruntu = image.copy()
    
    if method == 'basit':
        # Basit eşikleme - vektörel işlem
        binary = np.where(gri_goruntu >= threshold, 255, 0)
                
    elif method == 'otsu':
        # Otsu eşikleme
        # Histogram hesapla
        histogram = np.histogram(gri_goruntu, bins=256, range=(0, 256))[0]
        
        toplam_piksel = gri_goruntu.size
        toplam_deger = sum(i * h for i, h in enumerate(histogram))
        
        en_iyi_esik = 0
        en_iyi_varyans = 0
        agirlikli_toplam = 0
        piksel_sayisi = 0
        
        # Her olası eşik değeri için varyansı hesapla
        for esik in range(256):
            piksel_sayisi += histogram[esik]
            if piksel_sayisi == 0:
                continue
                
            agirlikli_toplam += esik * histogram[esik]
            
            w1 = piksel_sayisi / toplam_piksel
            w2 = 1 - w1
            
            if w2 == 0:
                break
                
            mu1 = agirlikli_toplam / piksel_sayisi
            mu2 = (toplam_deger - agirlikli_toplam) / (toplam_piksel - piksel_sayisi)
            
            # Sınıflar arası varyans
            varyans = w1 * w2 * (mu1 - mu2) ** 2
            
            if varyans > en_iyi_varyans:
                en_iyi_varyans = varyans
                en_iyi_esik = esik
        
        # Otsu eşik değeri ile binary dönüşüm - vektörel işlem
        binary = np.where(gri_goruntu >= en_iyi_esik, 255, 0)
                
    elif method == 'adaptif':
        # Adaptif eşikleme
        binary = adaptif_esikleme(gri_goruntu, blok_boyutu)
    
    else:
        raise ValueError("Geçersiz eşikleme yöntemi. Desteklenen yöntemler: 'basit', 'otsu', 'adaptif'")
    
    return binary.astype(np.uint8) 