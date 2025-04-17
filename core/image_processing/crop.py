import numpy as np

def goruntu_kirp(image: np.ndarray, start_x: int, start_y: int, width: int, height: int, aspect_ratio: bool = False) -> np.ndarray:
    """
    Görüntüyü belirtilen koordinatlardan başlayarak belirtilen boyutlarda kırpar.
    
    Args:
        image (np.ndarray): Giriş görüntüsü
        start_x (int): Başlangıç X koordinatı
        start_y (int): Başlangıç Y koordinatı
        width (int): Kırpılacak genişlik
        height (int): Kırpılacak yükseklik
        aspect_ratio (bool): En-boy oranını koru
        
    Returns:
        np.ndarray: Kırpılmış görüntü
    """
    # Görüntü boyutlarını al
    img_height, img_width = image.shape[:2]
    
    # Koordinatları sınırla
    start_x = max(0, min(start_x, img_width - 1))
    start_y = max(0, min(start_y, img_height - 1))
    
    # En-boy oranını koruma kontrolü
    if aspect_ratio and width > 0 and height > 0:
        # Orijinal görüntünün en-boy oranı
        original_ratio = img_width / img_height
        # Yeni en-boy oranı
        new_ratio = width / height
        
        # En-boy oranını koruyarak yeni boyutları hesapla
        if new_ratio > original_ratio:
            # Genişliği ayarla
            width = int(height * original_ratio)
        else:
            # Yüksekliği ayarla
            height = int(width / original_ratio)
    
    # Kırpma boyutlarını sınırla
    width = max(1, min(width, img_width - start_x))
    height = max(1, min(height, img_height - start_y))
    
    # Bitiş koordinatlarını hesapla
    end_x = start_x + width
    end_y = start_y + height
    
    # Görüntüyü kırp
    if len(image.shape) > 2:
        # RGB görüntü
        cropped = np.zeros((height, width, image.shape[2]), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                if 0 <= start_y + i < img_height and 0 <= start_x + j < img_width:
                    cropped[i, j] = image[start_y + i, start_x + j]
    else:
        # Gri tonlamalı görüntü
        cropped = np.zeros((height, width), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                if 0 <= start_y + i < img_height and 0 <= start_x + j < img_width:
                    cropped[i, j] = image[start_y + i, start_x + j]
    
    return cropped 