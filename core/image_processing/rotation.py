import numpy as np
from math import cos, sin, radians

def bilinear_interpolasyon(img: np.ndarray, x: float, y: float) -> np.ndarray:
    """
    Bilinear interpolasyon yöntemi ile piksel değerini hesaplar.
    """
    if x < 0 or y < 0 or x >= img.shape[1] - 1 or y >= img.shape[0] - 1:
        return np.zeros(3 if len(img.shape) > 2 else 1, dtype=np.uint8)
        
    x1, y1 = int(x), int(y)
    x2, y2 = x1 + 1, y1 + 1
    
    # Kesirli kısımları hesapla
    dx = x - x1
    dy = y - y1
    
    # Dört komşu pikselin değerlerini al ve interpolasyon uygula
    q11 = img[y1, x1].astype(float)[:3] if len(img.shape) > 2 else img[y1, x1].astype(float)
    q21 = img[y1, x2].astype(float)[:3] if len(img.shape) > 2 else img[y1, x2].astype(float)
    q12 = img[y2, x1].astype(float)[:3] if len(img.shape) > 2 else img[y2, x1].astype(float)
    q22 = img[y2, x2].astype(float)[:3] if len(img.shape) > 2 else img[y2, x2].astype(float)
    
    return np.uint8(
        (1 - dx) * (1 - dy) * q11 +
        dx * (1 - dy) * q21 +
        (1 - dx) * dy * q12 +
        dx * dy * q22
    )

def en_yakin_komsu(img: np.ndarray, x: float, y: float) -> np.ndarray:
    """
    En yakın komşu interpolasyon yöntemi ile piksel değerini hesaplar.
    """
    if x < 0 or y < 0 or x >= img.shape[1] or y >= img.shape[0]:
        return np.zeros(3 if len(img.shape) > 2 else 1, dtype=np.uint8)
    
    pixel = img[int(round(y)), int(round(x))]
    return pixel[:3] if len(img.shape) > 2 else pixel

def bikubik_interpolasyon(img: np.ndarray, x: float, y: float) -> np.ndarray:
    """
    Bikübik interpolasyon yöntemi ile piksel değerini hesaplar.
    Basitleştirilmiş versiyon - 2x2 pencere kullanır
    """
    if x < 0 or y < 0 or x >= img.shape[1] - 1 or y >= img.shape[0] - 1:
        return np.zeros(3 if len(img.shape) > 2 else 1, dtype=np.uint8)
        
    x1, y1 = int(x), int(y)
    x2, y2 = x1 + 1, y1 + 1
    
    dx = x - x1
    dy = y - y1
    
    # Kübik ağırlık fonksiyonu
    def cubic_weight(t):
        return 1 - 2 * t * t + t * t * t

    wx = cubic_weight(dx)
    wy = cubic_weight(dy)
    
    # 2x2 pencere için interpolasyon
    q11 = img[y1, x1].astype(float)[:3] if len(img.shape) > 2 else img[y1, x1].astype(float)
    q21 = img[y1, x2].astype(float)[:3] if len(img.shape) > 2 else img[y1, x2].astype(float)
    q12 = img[y2, x1].astype(float)[:3] if len(img.shape) > 2 else img[y2, x1].astype(float)
    q22 = img[y2, x2].astype(float)[:3] if len(img.shape) > 2 else img[y2, x2].astype(float)
    
    return np.uint8(
        wx * wy * q11 +
        (1 - wx) * wy * q21 +
        wx * (1 - wy) * q12 +
        (1 - wx) * (1 - wy) * q22
    )

def goruntu_dondur(image: np.ndarray, aci: float, interpolasyon: str = 'bilinear', boyut_koru: bool = True) -> np.ndarray:
    """
    Görüntüyü belirtilen açı kadar döndürür.
    
    Args:
        image (np.ndarray): Giriş görüntüsü
        aci (float): Döndürme açısı (derece)
        interpolasyon (str): Interpolasyon yöntemi ('bilinear', 'nearest', 'bicubic')
        boyut_koru (bool): True ise orijinal boyut korunur
        
    Returns:
        np.ndarray: Döndürülmüş görüntü
    """
    # Debug için giriş parametrelerini yazdır
    print(f"Döndürme fonksiyonu parametreleri:")
    print(f"Görüntü boyutu: {image.shape}")
    print(f"Açı: {aci}")
    print(f"İnterpolasyon: {interpolasyon}")
    print(f"Boyut korunacak mı: {boyut_koru}")
    
    # RGBA görüntüyü RGB'ye dönüştür
    if len(image.shape) > 2 and image.shape[2] == 4:
        image = image[:, :, :3]
    
    # Radyana çevir
    theta = np.radians(aci)
    
    height, width = image.shape[:2]
    is_rgb = len(image.shape) > 2
    
    # Döndürme matrisi
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    if boyut_koru:
        # Orijinal boyutları koru
        new_width = width
        new_height = height
    else:
        # Yeni boyutları hesapla
        new_width = int(abs(width * cos_theta) + abs(height * sin_theta))
        new_height = int(abs(height * cos_theta) + abs(width * sin_theta))
    
    # Merkez noktaları
    center_x = width / 2
    center_y = height / 2
    new_center_x = new_width / 2
    new_center_y = new_height / 2
    
    # Çıktı görüntüsü oluştur
    output = np.zeros((new_height, new_width, 3) if is_rgb else (new_height, new_width), dtype=np.uint8)
    
    # İnterpolasyon fonksiyonunu seç
    if interpolasyon == 'nearest':
        interp_func = en_yakin_komsu
    elif interpolasyon == 'bicubic':
        interp_func = bikubik_interpolasyon
    else:  # bilinear
        interp_func = bilinear_interpolasyon
    
    # Vektörize edilmiş koordinat dönüşümü
    y_coords, x_coords = np.mgrid[:new_height, :new_width]
    
    # Ters dönüşüm koordinatları
    src_x = cos_theta * (x_coords - new_center_x) - sin_theta * (y_coords - new_center_y) + center_x
    src_y = sin_theta * (x_coords - new_center_x) + cos_theta * (y_coords - new_center_y) + center_y
    
    # Geçerli koordinatları bul
    valid_coords = (src_x >= 0) & (src_x < width - 1) & (src_y >= 0) & (src_y < height - 1)
    y_valid, x_valid = np.where(valid_coords)
    
    # Debug için işlem bilgilerini yazdır
    print(f"Geçerli koordinat sayısı: {len(y_valid)}")
    print(f"Çıktı görüntü boyutu: {output.shape}")
    
    # Sadece geçerli koordinatlar için interpolasyon uygula
    for i in range(len(y_valid)):
        y, x = y_valid[i], x_valid[i]
        output[y, x] = interp_func(image, src_x[y, x], src_y[y, x])
    
    return output 