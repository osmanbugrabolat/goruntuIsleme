import numpy as np
from typing import Union, Tuple

def dogrusal_kontrast(image: np.ndarray, kontrast_faktor: float, kanal_bazli: bool = False) -> np.ndarray:
    """
    Doğrusal kontrast ayarlama işlemi uygular.
    
    Parametreler:
    image: Giriş görüntüsü (uint8)
    kontrast_faktor: Kontrast artırma/azaltma faktörü (0.1 ile 3.0 arası)
    kanal_bazli: RGB görüntüler için kanalları ayrı ayrı işle
    """
    # Görüntüyü float32'ye dönüştür
    image = image.astype(np.float32)
    
    if len(image.shape) == 3 and kanal_bazli:  # RGB görüntü, kanal bazlı işlem
        sonuc = np.zeros_like(image)
        for i in range(3):
            kanal = image[:,:,i]
            # Ortalama değeri hesapla
            ortalama = np.mean(kanal)
            # Kontrast ayarlama formülü: (pixel - ortalama) * kontrast + ortalama
            sonuc[:,:,i] = np.clip((kanal - ortalama) * kontrast_faktor + ortalama, 0, 255)
    else:
        if len(image.shape) == 3:  # RGB görüntü, tek kanal olarak işle
            ortalama = np.mean(image)
            sonuc = np.clip((image - ortalama) * kontrast_faktor + ortalama, 0, 255)
        else:  # Gri görüntü
            ortalama = np.mean(image)
            sonuc = np.clip((image - ortalama) * kontrast_faktor + ortalama, 0, 255)
    
    return sonuc.astype(np.uint8)

def gamma_duzeltme(image: np.ndarray, gamma: float, kanal_bazli: bool = False) -> np.ndarray:
    """
    Gamma düzeltmesi uygular.
    
    Parametreler:
    image: Giriş görüntüsü (uint8)
    gamma: Gamma değeri (0.1 ile 5.0 arası)
    kanal_bazli: RGB görüntüler için kanalları ayrı ayrı işle
    """
    # Görüntüyü 0-1 aralığına normalize et
    image = image.astype(np.float32) / 255.0
    
    if len(image.shape) == 3 and kanal_bazli:  # RGB görüntü, kanal bazlı işlem
        sonuc = np.zeros_like(image)
        for i in range(3):
            kanal = image[:,:,i]
            # Gamma düzeltme formülü: pixel ^ (1/gamma)
            sonuc[:,:,i] = np.power(kanal, 1.0/gamma)
    else:
        if len(image.shape) == 3:  # RGB görüntü, tek kanal olarak işle
            sonuc = np.power(image, 1.0/gamma)
        else:  # Gri görüntü
            sonuc = np.power(image, 1.0/gamma)
    
    # 0-255 aralığına dönüştür
    return np.clip(sonuc * 255.0, 0, 255).astype(np.uint8)

def logaritmik_donusum(image: np.ndarray, kanal_bazli: bool = False) -> np.ndarray:
    """
    Logaritmik dönüşüm uygular.
    
    Parametreler:
    image: Giriş görüntüsü (uint8)
    kanal_bazli: RGB görüntüler için kanalları ayrı ayrı işle
    """
    # Görüntüyü float32'ye dönüştür
    image = image.astype(np.float32)
    
    if len(image.shape) == 3 and kanal_bazli:  # RGB görüntü, kanal bazlı işlem
        sonuc = np.zeros_like(image)
        for i in range(3):
            kanal = image[:,:,i]
            # Maksimum değeri bul
            c = 255.0 / np.log(1 + np.max(kanal))
            # Logaritmik dönüşüm formülü: c * log(1 + pixel)
            sonuc[:,:,i] = c * np.log(1 + kanal)
    else:
        if len(image.shape) == 3:  # RGB görüntü, tek kanal olarak işle
            c = 255.0 / np.log(1 + np.max(image))
            sonuc = c * np.log(1 + image)
        else:  # Gri görüntü
            c = 255.0 / np.log(1 + np.max(image))
            sonuc = c * np.log(1 + image)
    
    return np.clip(sonuc, 0, 255).astype(np.uint8)

def kontrast_isle(image: np.ndarray, yontem: str, 
                 kontrast_faktor: float = 1.0, 
                 gamma: float = 1.0,
                 kanal_bazli: bool = False) -> np.ndarray:
    """
    Kontrast ayarlama işlemlerini uygular.
    
    Parametreler:
    image: Giriş görüntüsü
    yontem: İşlem yöntemi ('dogrusal', 'gamma', 'logaritmik')
    kontrast_faktor: Doğrusal kontrast için faktör değeri
    gamma: Gamma düzeltmesi için gamma değeri
    kanal_bazli: RGB görüntüler için kanalları ayrı ayrı işle
    """
    if yontem == 'dogrusal':
        return dogrusal_kontrast(image, kontrast_faktor, kanal_bazli)
    elif yontem == 'gamma':
        return gamma_duzeltme(image, gamma, kanal_bazli)
    elif yontem == 'logaritmik':
        return logaritmik_donusum(image, kanal_bazli)
    else:
        raise ValueError(f"Geçersiz kontrast ayarlama yöntemi: {yontem}") 