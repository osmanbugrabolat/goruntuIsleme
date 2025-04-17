import numpy as np

def rgb_to_gray(image: np.ndarray, method: str = 'ortalama') -> np.ndarray:
    """
    RGB görüntüyü gri seviye görüntüye dönüştürür.
    Hiçbir hazır fonksiyon kullanılmadan implementasyon.
    
    Args:
        image (np.ndarray): RGB görüntü (HxWx3)
        method (str): Dönüşüm yöntemi ('ortalama', 'agirlikli', 'luminosity')
    
    Returns:
        np.ndarray: Gri seviye görüntü (HxW)
    """
    # Görüntünün boyutlarını kontrol et
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Girdi görüntüsü RGB (HxWx3) formatında olmalıdır.")
    
    # Görüntü boyutlarını al
    height, width = image.shape[:2]
    
    # Çıktı görüntüsünü oluştur
    gray = np.zeros((height, width), dtype=np.uint8)
    
    # Her piksel için gri dönüşümü uygula
    for i in range(height):
        for j in range(width):
            r, g, b = image[i, j]
            
            if method == 'ortalama':
                # R, G, B kanallarının ortalamasını al
                gray_value = (int(r) + int(g) + int(b)) // 3
                
            elif method == 'agirlikli':
                # İnsan gözünün renk hassasiyetine göre ağırlıklandırma
                # R: %30, G: %59, B: %11
                gray_value = int(0.30 * r + 0.59 * g + 0.11 * b)
                
            elif method == 'luminosity':
                # Luminosity yöntemi - HSL renk uzayındaki L değeri
                # L = (max(R,G,B) + min(R,G,B)) / 2
                max_val = max(r, g, b)
                min_val = min(r, g, b)
                gray_value = (max_val + min_val) // 2
                
            else:
                raise ValueError("Geçersiz dönüşüm yöntemi. Desteklenen yöntemler: 'ortalama', 'agirlikli', 'luminosity'")
            
            # Değeri 0-255 aralığında sınırla
            gray_value = min(255, max(0, gray_value))
            gray[i, j] = gray_value
    
    return gray 