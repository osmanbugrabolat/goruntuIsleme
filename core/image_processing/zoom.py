import numpy as np
from math import floor, ceil, exp, pi

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
    q11 = img[y1, x1].astype(float)
    q21 = img[y1, x2].astype(float)
    q12 = img[y2, x1].astype(float)
    q22 = img[y2, x2].astype(float)
    
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
    
    return img[int(round(y)), int(round(x))]

def bikubik_kernel(x: float) -> float:
    """
    Bikübik interpolasyon için kernel fonksiyonu
    """
    x = abs(x)
    if x <= 1:
        return 1 - 2*x*x + x*x*x
    elif x < 2:
        return 4 - 8*x + 5*x*x - x*x*x
    return 0

def bikubik_interpolasyon(img: np.ndarray, x: float, y: float) -> np.ndarray:
    """
    Bikübik interpolasyon yöntemi ile piksel değerini hesaplar.
    """
    if x < 1 or y < 1 or x >= img.shape[1] - 2 or y >= img.shape[0] - 2:
        return bilinear_interpolasyon(img, x, y)
    
    x_floor, y_floor = floor(x), floor(y)
    result = np.zeros(3 if len(img.shape) > 2 else 1, dtype=float)
    
    for i in range(-1, 3):
        for j in range(-1, 3):
            weight = bikubik_kernel(x - (x_floor + j)) * bikubik_kernel(y - (y_floor + i))
            pixel = img[y_floor + i, x_floor + j].astype(float)
            result += weight * pixel
    
    return np.uint8(np.clip(result, 0, 255))

def lanczos_kernel(x: float, a: int = 3) -> float:
    """
    Lanczos interpolasyon için kernel fonksiyonu
    """
    if x == 0:
        return 1
    elif -a < x < a:
        return a * np.sin(np.pi * x) * np.sin(np.pi * x / a) / (np.pi * np.pi * x * x)
    return 0

def lanczos_interpolasyon(img: np.ndarray, x: float, y: float) -> np.ndarray:
    """
    Lanczos interpolasyon yöntemi ile piksel değerini hesaplar.
    """
    if x < 2 or y < 2 or x >= img.shape[1] - 3 or y >= img.shape[0] - 3:
        return bilinear_interpolasyon(img, x, y)
    
    x_floor, y_floor = floor(x), floor(y)
    result = np.zeros(3 if len(img.shape) > 2 else 1, dtype=float)
    
    for i in range(-2, 4):
        for j in range(-2, 4):
            weight = lanczos_kernel(x - (x_floor + j)) * lanczos_kernel(y - (y_floor + i))
            pixel = img[y_floor + i, x_floor + j].astype(float)
            result += weight * pixel
    
    return np.uint8(np.clip(result, 0, 255))

def gaussian_kernel(size, sigma):
    """
    Gaussian kernel oluşturur
    """
    kernel = np.zeros((size, size))
    center = size // 2
    
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = exp(-(x*x + y*y)/(2*sigma*sigma)) / (2*pi*sigma*sigma)
    
    return kernel / kernel.sum()

def gaussian_blur(image, sigma):
    """
    Gaussian bulanıklaştırma uygular
    """
    if sigma <= 0:
        return image
        
    # Kernel boyutunu hesapla
    kernel_size = max(3, int(2 * ceil(3*sigma)) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    # Kernel oluştur
    kernel = gaussian_kernel(kernel_size, sigma)
    
    # Görüntüyü genişlet
    pad = kernel_size // 2
    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) > 2 else 1
    
    if channels == 1:
        padded = np.pad(image, ((pad, pad), (pad, pad)), mode='reflect')
        output = np.zeros_like(image)
        
        for y in range(height):
            for x in range(width):
                window = padded[y:y+kernel_size, x:x+kernel_size]
                output[y, x] = np.sum(window * kernel)
    else:
        padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
        output = np.zeros_like(image)
        
        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    window = padded[y:y+kernel_size, x:x+kernel_size, c]
                    output[y, x, c] = np.sum(window * kernel)
    
    return output.astype(np.uint8)

def en_yakin_komsu_interpolasyon(goruntu, yeni_boyut):
    """
    En yakın komşu interpolasyonu ile görüntüyü yeniden boyutlandırır.
    """
    h, w = goruntu.shape[:2]
    yeni_h, yeni_w = yeni_boyut
    
    # Ölçekleme faktörleri
    x_scale = float(w - 1) / (yeni_w - 1) if yeni_w > 1 else 0
    y_scale = float(h - 1) / (yeni_h - 1) if yeni_h > 1 else 0
    
    # Yeni görüntü oluştur
    yeni_goruntu = np.zeros((yeni_h, yeni_w) + goruntu.shape[2:], dtype=goruntu.dtype)
    
    for i in range(yeni_h):
        for j in range(yeni_w):
            # Orijinal görüntüdeki koordinatları hesapla
            src_x = min(w - 1, round(j * x_scale))
            src_y = min(h - 1, round(i * y_scale))
            yeni_goruntu[i, j] = goruntu[src_y, src_x]
    
    return yeni_goruntu

def bilineer_interpolasyon(goruntu, yeni_boyut):
    """
    Bilineer interpolasyon ile görüntüyü yeniden boyutlandırır.
    """
    h, w = goruntu.shape[:2]
    yeni_h, yeni_w = yeni_boyut
    
    # Ölçekleme faktörleri
    x_scale = float(w - 1) / (yeni_w - 1) if yeni_w > 1 else 0
    y_scale = float(h - 1) / (yeni_h - 1) if yeni_h > 1 else 0
    
    # Yeni görüntü oluştur
    yeni_goruntu = np.zeros((yeni_h, yeni_w) + goruntu.shape[2:], dtype=np.float32)
    
    for i in range(yeni_h):
        for j in range(yeni_w):
            # Orijinal görüntüdeki koordinatları hesapla
            x = j * x_scale
            y = i * y_scale
            
            # En yakın 4 pikselin koordinatlarını bul
            x0 = int(np.floor(x))
            x1 = min(x0 + 1, w - 1)
            y0 = int(np.floor(y))
            y1 = min(y0 + 1, h - 1)
            
            # Kesirli kısımları hesapla
            dx = x - x0
            dy = y - y0
            
            # Bilineer interpolasyon
            piksel = (1 - dx) * (1 - dy) * goruntu[y0, x0].astype(np.float32) + \
                    dx * (1 - dy) * goruntu[y0, x1].astype(np.float32) + \
                    (1 - dx) * dy * goruntu[y1, x0].astype(np.float32) + \
                    dx * dy * goruntu[y1, x1].astype(np.float32)
            
            yeni_goruntu[i, j] = piksel
    
    return np.clip(yeni_goruntu, 0, 255).astype(np.uint8)

def bikubik_interpolasyon(goruntu, yeni_boyut):
    """
    Bikübik interpolasyon ile görüntüyü yeniden boyutlandırır.
    """
    def cubic(x):
        absx = abs(x)
        if absx <= 1:
            return 1.5 * absx**3 - 2.5 * absx**2 + 1
        elif absx < 2:
            return -0.5 * absx**3 + 2.5 * absx**2 - 4 * absx + 2
        return 0

    h, w = goruntu.shape[:2]
    yeni_h, yeni_w = yeni_boyut
    
    # Ölçekleme faktörleri
    x_scale = float(w - 1) / (yeni_w - 1) if yeni_w > 1 else 0
    y_scale = float(h - 1) / (yeni_h - 1) if yeni_h > 1 else 0
    
    # Yeni görüntü oluştur
    yeni_goruntu = np.zeros((yeni_h, yeni_w) + goruntu.shape[2:], dtype=goruntu.dtype)
    
    for i in range(yeni_h):
        for j in range(yeni_w):
            x = j * x_scale
            y = i * y_scale
            
            x_int = int(x)
            y_int = int(y)
            
            # 4x4 komşuluk için ağırlıkları hesapla
            for m in range(-1, 3):
                for n in range(-1, 3):
                    # Sınırları kontrol et
                    xi = max(0, min(w-1, x_int + m))
                    yi = max(0, min(h-1, y_int + n))
                    
                    # Kübik ağırlıkları hesapla
                    wx = cubic(x - (x_int + m))
                    wy = cubic(y - (y_int + n))
                    
                    # Piksel değerini güncelle
                    yeni_goruntu[i, j] += goruntu[yi, xi] * wx * wy
    
    return np.clip(yeni_goruntu, 0, 255).astype(np.uint8)

def lanczos_interpolasyon(goruntu, yeni_boyut, a=3):
    """
    Lanczos interpolasyonu ile görüntüyü yeniden boyutlandırır.
    """
    def lanczos(x, a):
        if x == 0:
            return 1
        elif -a < x < a:
            return a * np.sin(np.pi * x) * np.sin(np.pi * x / a) / (np.pi * np.pi * x * x)
        return 0

    h, w = goruntu.shape[:2]
    yeni_h, yeni_w = yeni_boyut
    
    # Ölçekleme faktörleri
    x_scale = float(w - 1) / (yeni_w - 1) if yeni_w > 1 else 0
    y_scale = float(h - 1) / (yeni_h - 1) if yeni_h > 1 else 0
    
    # Yeni görüntü oluştur
    yeni_goruntu = np.zeros((yeni_h, yeni_w) + goruntu.shape[2:], dtype=np.float32)
    
    for i in range(yeni_h):
        for j in range(yeni_w):
            x = j * x_scale
            y = i * y_scale
            
            x_int = int(x)
            y_int = int(y)
            
            # Lanczos kernel boyutu için komşuluğu hesapla
            for m in range(-a+1, a+1):
                for n in range(-a+1, a+1):
                    # Sınırları kontrol et
                    xi = max(0, min(w-1, x_int + m))
                    yi = max(0, min(h-1, y_int + n))
                    
                    # Lanczos ağırlıklarını hesapla
                    wx = lanczos(x - (x_int + m), a)
                    wy = lanczos(y - (y_int + n), a)
                    
                    # Piksel değerini güncelle
                    yeni_goruntu[i, j] += goruntu[yi, xi] * wx * wy
    
    return np.clip(yeni_goruntu, 0, 255).astype(np.uint8)

def goruntu_olcekle(image, olcek_faktoru, interpolasyon='linear', anti_aliasing=True):
    """
    Görüntüyü verilen ölçek faktörüne göre yeniden boyutlandırır.
    
    Parametreler:
    image: numpy.ndarray - Giriş görüntüsü
    olcek_faktoru: float - Ölçek faktörü (1.0 = orijinal boyut)
    interpolasyon: str - İnterpolasyon yöntemi ('nearest', 'linear', 'cubic', 'lanczos')
    anti_aliasing: bool - Küçültme işleminde anti-aliasing uygulansın mı?
    """
    if image is None:
        raise ValueError("Görüntü yüklenemedi")
    
    # Görüntü boyutlarını kontrol et
    if len(image.shape) < 2:
        raise ValueError("Geçersiz görüntü boyutları")
    
    # Ölçek faktörünü kontrol et ve dönüştür
    try:
        olcek = float(olcek_faktoru)
        print(f"Alınan ölçek faktörü: {olcek}")
        
        if not isinstance(olcek, (int, float)):
            raise ValueError("Ölçek faktörü sayısal bir değer olmalıdır")
            
        if olcek <= 0.0:
            raise ValueError("Ölçek faktörü pozitif olmalıdır")
            
        if olcek > 5.0:
            raise ValueError("Ölçek faktörü 5.0'dan büyük olamaz")
            
    except (TypeError, ValueError) as e:
        print(f"Ölçek faktörü hatası: {str(e)}")
        raise ValueError(f"Geçersiz ölçek faktörü: {str(e)}")
    
    print(f"Ölçekleme başlıyor: faktör={olcek}, interpolasyon={interpolasyon}, anti_aliasing={anti_aliasing}")
    print(f"Orijinal boyutlar: {image.shape}")
    
    # Yeni boyutları hesapla
    h, w = image.shape[:2]
    yeni_h = max(1, int(round(h * olcek)))
    yeni_w = max(1, int(round(w * olcek)))
    
    print(f"Hesaplanan yeni boyutlar: {yeni_h}x{yeni_w}")
    
    # Boyut kontrolü
    max_boyut = 4000  # Maksimum izin verilen boyut
    if yeni_h > max_boyut or yeni_w > max_boyut:
        raise ValueError(f"Yeni boyutlar çok büyük. Maksimum {max_boyut}x{max_boyut} piksel destekleniyor.")
    
    # Boyut değişmediyse orijinal görüntüyü döndür
    if yeni_h == h and yeni_w == w:
        return image.copy()
    
    # Küçültme işleminde anti-aliasing uygula
    if anti_aliasing and olcek < 1.0:
        sigma = max(0, (1 / olcek - 1) * 0.5)
        image = gaussian_blur(image, sigma)
    
    # İnterpolasyon yöntemini seç ve uygula
    interpolasyon = interpolasyon.lower()
    try:
        if interpolasyon == 'nearest':
            sonuc = en_yakin_komsu_interpolasyon(image, (yeni_h, yeni_w))
        elif interpolasyon == 'linear':
            sonuc = bilineer_interpolasyon(image, (yeni_h, yeni_w))
        elif interpolasyon == 'cubic':
            sonuc = bikubik_interpolasyon(image, (yeni_h, yeni_w))
        elif interpolasyon == 'lanczos':
            sonuc = lanczos_interpolasyon(image, (yeni_h, yeni_w))
        else:
            print(f"Geçersiz interpolasyon yöntemi: {interpolasyon}, varsayılan olarak bilineer kullanılıyor")
            sonuc = bilineer_interpolasyon(image, (yeni_h, yeni_w))
        
        print(f"İşlenmiş görüntü boyutları: {sonuc.shape}")
        return sonuc
        
    except Exception as e:
        print(f"Ölçekleme hatası: {str(e)}")
        raise ValueError(f"Ölçekleme işlemi başarısız oldu: {str(e)}") 