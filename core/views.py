from django.shortcuts import render
from django.http import JsonResponse
import numpy as np
import base64
from PIL import Image
import io
from .image_processing.gray_scale import rgb_to_gray
from .image_processing.binary import binary_donusum as binary_donusum_isle
from io import BytesIO
from django.views.decorators.http import require_http_methods

def read_image_file(file) -> np.ndarray:
    """
    Yüklenen dosyayı numpy dizisine dönüştürür.
    Tüm yaygın görüntü formatlarını destekler.
    """
    try:
        # Dosyayı byte olarak oku
        image_data = file.read()
        image_buffer = io.BytesIO(image_data)
        
        # PIL ile görüntüyü aç
        with Image.open(image_buffer) as img:
            # RGB moduna dönüştür
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Numpy dizisine dönüştür
            return np.array(img)
            
    except Exception as e:
        raise ValueError(f"Görüntü okunamadı: {str(e)}")

def encode_image_base64(image_data: np.ndarray) -> str:
    """
    Görüntü verisini base64 formatına dönüştürür.
    """
    try:
        # Numpy dizisini PIL görüntüsüne dönüştür
        image = Image.fromarray(image_data.astype('uint8'))
        
        # Görüntüyü JPEG olarak buffer'a kaydet
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        buffer.seek(0)
        
        # Base64'e dönüştür
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{image_base64}"
    
    except Exception as e:
        raise ValueError(f"Görüntü encode edilemedi: {str(e)}")

def gri_donusum(request):
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            # Gönderilen görüntüyü oku
            image_file = request.FILES['image']
            image_array = read_image_file(image_file)
            
            # Dönüşüm yöntemini al
            method = request.POST.get('method', 'ortalama')
            
            # Gri dönüşümü uygula
            gray_image = rgb_to_gray(image_array, method)
            
            # İşlenmiş görüntüyü base64 formatına dönüştür
            processed_image = encode_image_base64(gray_image)
            
            return JsonResponse({
                'success': True,
                'processed_image': processed_image
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return render(request, 'core/islemler/gri_donusum.html')

def anasayfa(request):
    return render(request, 'core/anasayfa.html')

@require_http_methods(["GET", "POST"])
def binary_donusum(request):
    if request.method == "POST":
        try:
            # Gelen görüntüyü al
            image_data = request.FILES.get('image')
            if not image_data:
                return JsonResponse({'error': 'Görüntü bulunamadı'}, status=400)
            
            # Parametreleri al
            threshold = int(request.POST.get('esikDegeri', 127))
            method = request.POST.get('esiklemeYontemi', 'basit')
            
            # Debug için parametre değerlerini yazdır
            print(f"Alınan parametreler: threshold={threshold}, method={method}")
            
            # Görüntüyü numpy dizisine çevir
            image_array = read_image_file(image_data)
            
            # Binary dönüşümü uygula
            processed_image = binary_donusum_isle(
                image=image_array,
                threshold=threshold,
                method=method
            )
            
            # Sonuç görüntüsünü base64'e çevir
            result_image = Image.fromarray(processed_image)
            buffer = BytesIO()
            result_image.save(buffer, format='PNG')
            base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return JsonResponse({
                'processed_image': f'data:image/png;base64,{base64_image}',
                'message': 'Binary dönüşüm başarıyla uygulandı'
            })
            
        except Exception as e:
            print(f"Hata oluştu: {str(e)}")  # Hata mesajını sunucu konsoluna yazdır
            return JsonResponse({'error': str(e)}, status=400)
    
    return render(request, 'core/islemler/binary_donusum.html')

def goruntu_dondurme(request):
    return render(request, 'core/islemler/goruntu_dondurme.html')

def goruntu_kirpma(request):
    return render(request, 'core/islemler/goruntu_kirpma.html')

def yakinlastirma(request):
    return render(request, 'core/islemler/yakinlastirma.html')

def histogram(request):
    return render(request, 'core/islemler/histogram.html')

def kontrast(request):
    return render(request, 'core/islemler/kontrast.html')

def kenar_bulma(request):
    return render(request, 'core/islemler/kenar_algilama.html')

def gurultu(request):
    return render(request, 'core/islemler/gurultu.html')

def morfolojik(request):
    return render(request, 'core/islemler/morfolojik.html')
