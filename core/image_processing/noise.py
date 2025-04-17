import numpy as np
from typing import Union, Tuple, Literal

def tuz_biber_gurultu(image: np.ndarray, yogunluk: float) -> np.ndarray:
    """
    Görüntüye tuz ve biber gürültüsü ekler.
    
    Parametreler:
    image: Giriş görüntüsü (uint8)
    yogunluk: Gürültü yoğunluğu (0-100 arası)
    """
    # Görüntüyü kopyala
    sonuc = image.copy()
    
    # Yoğunluğu 0-1 aralığına dönüştür
    prob = yogunluk / 100.0
    
    # Rastgele noktalar oluştur
    rng = np.random.default_rng()
    tuz = rng.random(image.shape[:2]) < (prob / 2)
    biber = rng.random(image.shape[:2]) < (prob / 2)
    
    # Görüntü tipini kontrol et
    if len(image.shape) == 3:  # Çok kanallı görüntü
        kanal_sayisi = image.shape[2]
        if kanal_sayisi == 3:  # RGB
            sonuc[tuz] = [255, 255, 255]
            sonuc[biber] = [0, 0, 0]
        elif kanal_sayisi == 4:  # RGBA
            sonuc[tuz] = [255, 255, 255, 255]
            sonuc[biber] = [0, 0, 0, 255]
    else:  # Gri görüntü
        sonuc[tuz] = 255
        sonuc[biber] = 0
        
    return sonuc

def gauss_gurultu(image: np.ndarray, yogunluk: float) -> np.ndarray:
    """
    Görüntüye Gauss gürültüsü ekler.
    
    Parametreler:
    image: Giriş görüntüsü (uint8)
    yogunluk: Gürültü yoğunluğu (0-100 arası)
    """
    # Görüntüyü float32'ye dönüştür
    sonuc = image.astype(np.float32)
    
    # Yoğunluğa göre standart sapma hesapla (0-50 arası)
    std = (yogunluk / 2)
    
    # Gauss gürültüsü oluştur
    rng = np.random.default_rng()
    gurultu = rng.normal(0, std, image.shape).astype(np.float32)
    
    # Gürültüyü ekle ve sınırla
    sonuc = np.clip(sonuc + gurultu, 0, 255)
    
    return sonuc.astype(np.uint8)

def benek_gurultu(image: np.ndarray, yogunluk: float) -> np.ndarray:
    """
    Görüntüye benek (speckle) gürültüsü ekler.
    
    Parametreler:
    image: Giriş görüntüsü (uint8)
    yogunluk: Gürültü yoğunluğu (0-100 arası)
    """
    # Görüntüyü float32'ye dönüştür
    sonuc = image.astype(np.float32)
    
    # Yoğunluğu 0-1 aralığına dönüştür
    yogunluk = yogunluk / 100.0
    
    # Benek gürültüsü oluştur
    rng = np.random.default_rng()
    gurultu = yogunluk * rng.normal(0, 1, image.shape)
    
    # Gürültüyü çarpımsal olarak ekle ve sınırla
    sonuc = np.clip(sonuc * (1 + gurultu), 0, 255)
    
    return sonuc.astype(np.uint8)

def ortalama_filtre(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Görüntüye ortalama filtre uygular.
    
    Parametreler:
    image: Giriş görüntüsü (uint8)
    kernel_size: Filtre boyutu (3, 5, 7, 9)
    """
    # Görüntüyü float32'ye dönüştür
    sonuc = image.astype(np.float32)
    
    # Padding ekle
    pad = kernel_size // 2
    if len(image.shape) == 3:  # Renkli görüntü
        padded = np.pad(sonuc, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    else:  # Gri görüntü
        padded = np.pad(sonuc, pad, mode='reflect')
    
    # Ortalama filtreyi uygula
    for i in range(pad, padded.shape[0] - pad):
        for j in range(pad, padded.shape[1] - pad):
            if len(image.shape) == 3:  # Renkli görüntü
                for k in range(3):  # Her kanal için
                    pencere = padded[i-pad:i+pad+1, j-pad:j+pad+1, k]
                    sonuc[i-pad, j-pad, k] = np.mean(pencere)
            else:  # Gri görüntü
                pencere = padded[i-pad:i+pad+1, j-pad:j+pad+1]
                sonuc[i-pad, j-pad] = np.mean(pencere)
    
    return sonuc.astype(np.uint8)

def medyan_filtre(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Görüntüye medyan filtre uygular.
    
    Parametreler:
    image: Giriş görüntüsü (uint8)
    kernel_size: Filtre boyutu (3, 5, 7, 9)
    """
    # Görüntüyü kopyala
    sonuc = image.copy()
    
    # Padding ekle
    pad = kernel_size // 2
    if len(image.shape) == 3:  # Renkli görüntü
        padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    else:  # Gri görüntü
        padded = np.pad(image, pad, mode='reflect')
    
    # Medyan filtreyi uygula
    for i in range(pad, padded.shape[0] - pad):
        for j in range(pad, padded.shape[1] - pad):
            if len(image.shape) == 3:  # Renkli görüntü
                for k in range(3):  # Her kanal için
                    pencere = padded[i-pad:i+pad+1, j-pad:j+pad+1, k]
                    sonuc[i-pad, j-pad, k] = np.median(pencere)
            else:  # Gri görüntü
                pencere = padded[i-pad:i+pad+1, j-pad:j+pad+1]
                sonuc[i-pad, j-pad] = np.median(pencere)
    
    return sonuc

def gauss_filtre(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Görüntüye Gauss filtresi uygular.
    
    Parametreler:
    image: Giriş görüntüsü (uint8)
    kernel_size: Filtre boyutu (3, 5, 7, 9)
    """
    # Görüntüyü float32'ye dönüştür
    sonuc = image.astype(np.float32)
    
    # Gauss çekirdeği oluştur
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    x = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    kern1d = np.exp(-0.5 * x ** 2 / sigma ** 2)
    kernel = kern1d[:, None] * kern1d[None, :]
    kernel = kernel / kernel.sum()  # Normalize et
    
    # Padding ekle
    pad = kernel_size // 2
    if len(image.shape) == 3:  # Renkli görüntü
        padded = np.pad(sonuc, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    else:  # Gri görüntü
        padded = np.pad(sonuc, pad, mode='reflect')
    
    # Gauss filtreyi uygula
    for i in range(pad, padded.shape[0] - pad):
        for j in range(pad, padded.shape[1] - pad):
            if len(image.shape) == 3:  # Renkli görüntü
                for k in range(3):  # Her kanal için
                    pencere = padded[i-pad:i+pad+1, j-pad:j+pad+1, k]
                    sonuc[i-pad, j-pad, k] = np.sum(pencere * kernel)
            else:  # Gri görüntü
                pencere = padded[i-pad:i+pad+1, j-pad:j+pad+1]
                sonuc[i-pad, j-pad] = np.sum(pencere * kernel)
    
    return sonuc.astype(np.uint8)

def gurultu_isle(
    image: np.ndarray,
    islem_turu: Literal['ekle', 'temizle'],
    gurultu_turu: str = 'saltpepper',
    filtre_turu: str = 'mean',
    yogunluk: float = 10.0,
    filtre_boyutu: int = 3
) -> np.ndarray:
    """
    Görüntüye gürültü ekleme veya temizleme işlemi uygular.
    
    Parametreler:
    image: Giriş görüntüsü
    islem_turu: İşlem türü ('ekle' veya 'temizle')
    gurultu_turu: Gürültü türü ('saltpepper', 'gaussian', 'speckle')
    filtre_turu: Filtre türü ('mean', 'median', 'gaussian')
    yogunluk: Gürültü yoğunluğu (0-100 arası)
    filtre_boyutu: Filtre boyutu (3, 5, 7, 9)
    """
    if islem_turu == 'ekle':
        if gurultu_turu == 'saltpepper':
            return tuz_biber_gurultu(image, yogunluk)
        elif gurultu_turu == 'gaussian':
            return gauss_gurultu(image, yogunluk)
        elif gurultu_turu == 'speckle':
            return benek_gurultu(image, yogunluk)
        else:
            raise ValueError(f"Geçersiz gürültü türü: {gurultu_turu}")
    
    elif islem_turu == 'temizle':
        if filtre_turu == 'mean':
            return ortalama_filtre(image, filtre_boyutu)
        elif filtre_turu == 'median':
            return medyan_filtre(image, filtre_boyutu)
        elif filtre_turu == 'gaussian':
            return gauss_filtre(image, filtre_boyutu)
        else:
            raise ValueError(f"Geçersiz filtre türü: {filtre_turu}")
    
    else:
        raise ValueError(f"Geçersiz işlem türü: {islem_turu}") 