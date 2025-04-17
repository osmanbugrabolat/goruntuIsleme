import numpy as np
from typing import Tuple, Literal

def yapisal_element_olustur(
    sekil: str,
    genislik: int,
    yukseklik: int
) -> np.ndarray:
    """
    Belirtilen şekil ve boyutlarda yapısal element oluşturur.
    
    Parametreler:
    sekil: Yapısal elementin şekli ('kare', 'dikdortgen', 'elips', 'cross')
    genislik: Yapısal elementin genişliği
    yukseklik: Yapısal elementin yüksekliği
    """
    # Merkez noktaları hesapla
    merkez_y = yukseklik // 2
    merkez_x = genislik // 2
    
    # Boş yapısal element oluştur
    element = np.zeros((yukseklik, genislik), dtype=np.uint8)
    
    if sekil == 'kare':
        element.fill(1)
    
    elif sekil == 'dikdortgen':
        element.fill(1)
    
    elif sekil == 'elips':
        y, x = np.ogrid[-merkez_y:yukseklik-merkez_y, -merkez_x:genislik-merkez_x]
        elips = (x*x)/(merkez_x*merkez_x) + (y*y)/(merkez_y*merkez_y) <= 1
        element[elips] = 1
    
    elif sekil == 'cross':
        element[merkez_y, :] = 1
        element[:, merkez_x] = 1
    
    return element

def genisletme(
    image: np.ndarray,
    yapisal_element: np.ndarray
) -> np.ndarray:
    """
    Görüntüye genişletme işlemi uygular.
    
    Parametreler:
    image: Giriş görüntüsü
    yapisal_element: Yapısal element
    """
    # Görüntüyü kopyala
    sonuc = np.zeros_like(image)
    
    # Yapısal element boyutlarını al
    se_height, se_width = yapisal_element.shape
    pad_h, pad_w = se_height // 2, se_width // 2
    
    # Padding ekle
    if len(image.shape) == 3:  # Renkli görüntü
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
    else:  # Gri görüntü
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    
    # Genişletme işlemi
    for i in range(pad_h, padded.shape[0] - pad_h):
        for j in range(pad_w, padded.shape[1] - pad_w):
            if len(image.shape) == 3:  # Renkli görüntü
                for k in range(3):  # Her kanal için
                    pencere = padded[i-pad_h:i+pad_h+1, j-pad_w:j+pad_w+1, k]
                    sonuc[i-pad_h, j-pad_w, k] = np.max(pencere[yapisal_element == 1])
            else:  # Gri görüntü
                pencere = padded[i-pad_h:i+pad_h+1, j-pad_w:j+pad_w+1]
                sonuc[i-pad_h, j-pad_w] = np.max(pencere[yapisal_element == 1])
    
    return sonuc

def erozyon(
    image: np.ndarray,
    yapisal_element: np.ndarray
) -> np.ndarray:
    """
    Görüntüye erozyon işlemi uygular.
    
    Parametreler:
    image: Giriş görüntüsü
    yapisal_element: Yapısal element
    """
    # Görüntüyü kopyala
    sonuc = np.zeros_like(image)
    
    # Yapısal element boyutlarını al
    se_height, se_width = yapisal_element.shape
    pad_h, pad_w = se_height // 2, se_width // 2
    
    # Padding ekle
    if len(image.shape) == 3:  # Renkli görüntü
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant', constant_values=255)
    else:  # Gri görüntü
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=255)
    
    # Erozyon işlemi
    for i in range(pad_h, padded.shape[0] - pad_h):
        for j in range(pad_w, padded.shape[1] - pad_w):
            if len(image.shape) == 3:  # Renkli görüntü
                for k in range(3):  # Her kanal için
                    pencere = padded[i-pad_h:i+pad_h+1, j-pad_w:j+pad_w+1, k]
                    sonuc[i-pad_h, j-pad_w, k] = np.min(pencere[yapisal_element == 1])
            else:  # Gri görüntü
                pencere = padded[i-pad_h:i+pad_h+1, j-pad_w:j+pad_w+1]
                sonuc[i-pad_h, j-pad_w] = np.min(pencere[yapisal_element == 1])
    
    return sonuc

def morfolojik_gradyan(
    image: np.ndarray,
    yapisal_element: np.ndarray
) -> np.ndarray:
    """
    Görüntüye morfolojik gradyan işlemi uygular.
    
    Parametreler:
    image: Giriş görüntüsü
    yapisal_element: Yapısal element
    """
    # Genişletme ve erozyon işlemlerini uygula
    genisletilmis = genisletme(image, yapisal_element)
    erozyonlu = erozyon(image, yapisal_element)
    
    # Gradyan = Genişletme - Erozyon
    return np.clip(genisletilmis.astype(np.int16) - erozyonlu.astype(np.int16), 0, 255).astype(np.uint8)

def morfolojik_isle(
    image: np.ndarray,
    islem_turu: str,
    yapisal_element_sekli: str,
    genislik: int,
    yukseklik: int,
    on_islem: bool = False
) -> np.ndarray:
    """
    Görüntüye morfolojik işlem uygular.
    
    Parametreler:
    image: Giriş görüntüsü
    islem_turu: İşlem türü ('genisletme', 'erozyon', 'acma', 'kapama', 'gradyan')
    yapisal_element_sekli: Yapısal element şekli ('kare', 'dikdortgen', 'elips', 'cross')
    genislik: Yapısal element genişliği
    yukseklik: Yapısal element yüksekliği
    on_islem: İkili görüntüye dönüştürme işlemi uygulanacak mı?
    """
    # Ön işlem (ikili görüntüye dönüştürme)
    if on_islem:
        if len(image.shape) == 3:  # Renkli görüntü
            image = np.mean(image, axis=2).astype(np.uint8)
        image = (image > 127).astype(np.uint8) * 255
    
    # Yapısal elementi oluştur
    yapisal_element = yapisal_element_olustur(yapisal_element_sekli, genislik, yukseklik)
    
    # İşlem türüne göre morfolojik işlemi uygula
    if islem_turu == 'genisletme':
        return genisletme(image, yapisal_element)
    
    elif islem_turu == 'erozyon':
        return erozyon(image, yapisal_element)
    
    elif islem_turu == 'acma':
        # Açma = Erozyon + Genişletme
        erozyonlu = erozyon(image, yapisal_element)
        return genisletme(erozyonlu, yapisal_element)
    
    elif islem_turu == 'kapama':
        # Kapama = Genişletme + Erozyon
        genisletilmis = genisletme(image, yapisal_element)
        return erozyon(genisletilmis, yapisal_element)
    
    elif islem_turu == 'gradyan':
        return morfolojik_gradyan(image, yapisal_element)
    
    else:
        raise ValueError(f"Geçersiz işlem türü: {islem_turu}") 