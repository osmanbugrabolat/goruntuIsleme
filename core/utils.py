import numpy as np

def gri_donustur(image_array):
    """Görüntüyü gri tonlamaya çevirir."""
    if len(image_array.shape) == 3:
        return np.mean(image_array, axis=2).astype(np.uint8)
    return image_array

def otsu_esikleme(gri_goruntu):
    """
    Otsu eşikleme yöntemi ile optimal eşik değerini bulur.
    """
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
            
    return en_iyi_esik

def adaptif_esikleme(gri_goruntu, blok_boyutu=11):
    """
    Adaptif eşikleme yöntemi uygular.
    Her piksel için çevresindeki piksellerin ortalamasını eşik değeri olarak kullanır.
    """
    if blok_boyutu % 2 == 0:
        blok_boyutu += 1  # Blok boyutu tek sayı olmalı
        
    padding = blok_boyutu // 2
    rows, cols = gri_goruntu.shape
    sonuc = np.zeros_like(gri_goruntu)
    
    # Görüntüyü padding ile genişlet
    padded = np.pad(gri_goruntu, padding, mode='edge')
    
    # Her piksel için lokal eşikleme uygula
    for i in range(rows):
        for j in range(cols):
            # Piksel çevresindeki bloğu al
            blok = padded[i:i+blok_boyutu, j:j+blok_boyutu]
            # Bloğun ortalamasını eşik değeri olarak kullan
            esik = np.mean(blok)
            # Eşikleme uygula
            sonuc[i, j] = 255 if gri_goruntu[i, j] >= esik else 0
            
    return sonuc

def binary_donusum(image_array, threshold=127, yontem='basit', blok_boyutu=11):
    """
    Görüntüyü binary formata dönüştürür.
    
    Args:
        image_array (numpy.ndarray): Giriş görüntüsü
        threshold (int): Eşik değeri (0-255 arası), sadece basit eşiklemede kullanılır
        yontem (str): Eşikleme yöntemi ('basit', 'otsu', 'adaptif')
        blok_boyutu (int): Adaptif eşikleme için blok boyutu
    
    Returns:
        numpy.ndarray: Binary formata dönüştürülmüş görüntü
    """
    # Görüntüyü gri tonlamaya çevir
    gri_goruntu = gri_donustur(image_array)
    
    if yontem == 'basit':
        # Basit eşikleme
        binary_goruntu = np.where(gri_goruntu >= threshold, 255, 0)
    elif yontem == 'otsu':
        # Otsu eşikleme
        otsu_esik = otsu_esikleme(gri_goruntu)
        binary_goruntu = np.where(gri_goruntu >= otsu_esik, 255, 0)
    elif yontem == 'adaptif':
        # Adaptif eşikleme
        binary_goruntu = adaptif_esikleme(gri_goruntu, blok_boyutu)
    else:
        raise ValueError("Geçersiz eşikleme yöntemi. Kullanılabilir yöntemler: 'basit', 'otsu', 'adaptif'")
    
    return binary_goruntu.astype(np.uint8) 