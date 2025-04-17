import numpy as np

def gaussian_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """Gaussian blur filtresi uygular"""
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Gaussian kernel oluştur
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    center = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size))
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    kernel = kernel / kernel.sum()
    
    # Görüntüye padding uygula
    pad_size = kernel_size // 2
    padded = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), mode='reflect')
    result = np.zeros_like(image, dtype=np.float32)
    
    # Konvolüsyon işlemi
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded[i:i+kernel_size, j:j+kernel_size]
            result[i, j] = np.sum(window * kernel)
    
    return result.astype(np.uint8)

def sobel_operator(image: np.ndarray, direction: str = 'her_iki') -> np.ndarray:
    """Sobel operatörü ile kenar tespiti yapar"""
    # Sobel kernelleri
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    # Görüntüye padding uygula
    padded = np.pad(image, ((1, 1), (1, 1)), mode='reflect')
    gradient_x = np.zeros_like(image, dtype=np.float32)
    gradient_y = np.zeros_like(image, dtype=np.float32)
    
    # Konvolüsyon işlemi
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if direction in ['yatay', 'her_iki']:
                window = padded[i:i+3, j:j+3]
                gradient_x[i, j] = np.abs(np.sum(window * sobel_x))
            
            if direction in ['dikey', 'her_iki']:
                window = padded[i:i+3, j:j+3]
                gradient_y[i, j] = np.abs(np.sum(window * sobel_y))
    
    if direction == 'yatay':
        return gradient_x.astype(np.uint8)
    elif direction == 'dikey':
        return gradient_y.astype(np.uint8)
    else:
        return np.sqrt(gradient_x**2 + gradient_y**2).astype(np.uint8)

def prewitt_operator(image: np.ndarray, direction: str = 'her_iki') -> np.ndarray:
    """Prewitt operatörü ile kenar tespiti yapar"""
    # Prewitt kernelleri
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    
    # Görüntüye padding uygula
    padded = np.pad(image, ((1, 1), (1, 1)), mode='reflect')
    gradient_x = np.zeros_like(image, dtype=np.float32)
    gradient_y = np.zeros_like(image, dtype=np.float32)
    
    # Konvolüsyon işlemi
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if direction in ['yatay', 'her_iki']:
                window = padded[i:i+3, j:j+3]
                gradient_x[i, j] = np.abs(np.sum(window * prewitt_x))
            
            if direction in ['dikey', 'her_iki']:
                window = padded[i:i+3, j:j+3]
                gradient_y[i, j] = np.abs(np.sum(window * prewitt_y))
    
    if direction == 'yatay':
        return gradient_x.astype(np.uint8)
    elif direction == 'dikey':
        return gradient_y.astype(np.uint8)
    else:
        return np.sqrt(gradient_x**2 + gradient_y**2).astype(np.uint8)

def laplacian_operator(image: np.ndarray) -> np.ndarray:
    """Laplacian operatörü ile kenar tespiti yapar"""
    # Laplacian kernel
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    
    # Görüntüye padding uygula
    padded = np.pad(image, ((1, 1), (1, 1)), mode='reflect')
    result = np.zeros_like(image, dtype=np.float32)
    
    # Konvolüsyon işlemi
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded[i:i+3, j:j+3]
            result[i, j] = np.abs(np.sum(window * kernel))
    
    return result.astype(np.uint8)

def non_maximum_suppression(gradient_magnitude: np.ndarray, gradient_direction: np.ndarray) -> np.ndarray:
    """Canny kenar tespiti için non-maximum suppression uygular"""
    result = np.zeros_like(gradient_magnitude)
    height, width = gradient_magnitude.shape
    
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            angle = gradient_direction[i, j]
            
            # 0 derece (yatay)
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                neighbors = [gradient_magnitude[i, j-1], gradient_magnitude[i, j+1]]
            # 45 derece
            elif 22.5 <= angle < 67.5:
                neighbors = [gradient_magnitude[i-1, j+1], gradient_magnitude[i+1, j-1]]
            # 90 derece (dikey)
            elif 67.5 <= angle < 112.5:
                neighbors = [gradient_magnitude[i-1, j], gradient_magnitude[i+1, j]]
            # 135 derece
            else:
                neighbors = [gradient_magnitude[i-1, j-1], gradient_magnitude[i+1, j+1]]
            
            if gradient_magnitude[i, j] >= max(neighbors):
                result[i, j] = gradient_magnitude[i, j]
    
    return result

def canny_edge_detection(image: np.ndarray, low_threshold: int, high_threshold: int) -> np.ndarray:
    """Canny kenar tespit algoritması"""
    # 1. Gaussian blur uygula
    blurred = gaussian_blur(image, 5)
    
    # 2. Gradyanları hesapla (Sobel)
    gradient_x = sobel_operator(blurred, 'yatay')
    gradient_y = sobel_operator(blurred, 'dikey')
    
    gradient_magnitude = np.sqrt(gradient_x.astype(np.float32)**2 + gradient_y.astype(np.float32)**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x) * 180 / np.pi
    gradient_direction[gradient_direction < 0] += 180
    
    # 3. Non-maximum suppression
    suppressed = non_maximum_suppression(gradient_magnitude, gradient_direction)
    
    # 4. Double thresholding
    strong_edges = (suppressed > high_threshold).astype(np.uint8) * 255
    weak_edges = ((suppressed >= low_threshold) & (suppressed <= high_threshold)).astype(np.uint8) * 128
    
    # 5. Edge tracking by hysteresis
    result = np.zeros_like(image)
    result[suppressed > high_threshold] = 255
    
    # Zayıf kenarları kontrol et
    height, width = image.shape
    for i in range(1, height-1):
        for j in range(1, width-1):
            if weak_edges[i, j] == 128:
                if np.any(strong_edges[i-1:i+2, j-1:j+2] == 255):
                    result[i, j] = 255
    
    return result

def kenar_algilama(image: np.ndarray, yontem: str, **kwargs) -> np.ndarray:
    """Ana kenar algılama fonksiyonu"""
    # Görüntüyü gri tonlamaya çevir
    if len(image.shape) == 3:
        image = np.mean(image, axis=2).astype(np.uint8)
    
    # Ön işlem uygula
    if kwargs.get('on_islem', False):
        filtre_boyutu = kwargs.get('filtre_boyutu', 3)
        image = gaussian_blur(image, filtre_boyutu)
    
    # Kenar algılama yöntemini uygula
    if yontem == 'sobel':
        return sobel_operator(image, kwargs.get('yon', 'her_iki'))
    elif yontem == 'prewitt':
        return prewitt_operator(image, kwargs.get('yon', 'her_iki'))
    elif yontem == 'laplacian':
        return laplacian_operator(image)
    elif yontem == 'canny':
        return canny_edge_detection(
            image,
            kwargs.get('alt_esik', 100),
            kwargs.get('ust_esik', 200)
        )
    else:
        raise ValueError(f"Geçersiz yöntem: {yontem}") 