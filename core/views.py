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
from django.views.decorators.csrf import csrf_exempt
import json
from .image_processing.rotation import goruntu_dondur
from .image_processing.crop import goruntu_kirp
from .image_processing.zoom import goruntu_olcekle
from django.conf import settings
from .image_processing.histogram import histogram_isle, histogram_hesapla, histogram_grafigi_olustur
from .image_processing.contrast import kontrast_isle

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

@csrf_exempt
def goruntu_dondurme_isle(request):
    if request.method == 'POST':
        try:
            # JSON verisini al
            data = json.loads(request.body)
            
            # Base64 görüntüyü numpy dizisine dönüştür
            image_data = base64.b64decode(data['image'].split(',')[1])
            image_buffer = BytesIO(image_data)
            image = Image.open(image_buffer)
            image_array = np.array(image)
            
            # Parametreleri al
            aci = float(data['aci'])
            interpolasyon = data['interpolasyon']
            boyut_koru = data['boyutKoru']
            
            # Görüntüyü döndür
            donmus_goruntu = goruntu_dondur(image_array, aci, interpolasyon, boyut_koru)
            
            # Sonuç görüntüsünü base64'e dönüştür
            result_image = Image.fromarray(donmus_goruntu)
            buffer = BytesIO()
            result_image.save(buffer, format='PNG')
            base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return JsonResponse({
                'success': True,
                'image': f'data:image/png;base64,{base64_image}'
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

@csrf_exempt
def goruntu_kirpma_isle(request):
    if request.method == 'POST':
        try:
            # JSON verisini al
            data = json.loads(request.body)
            
            # Base64 görüntüyü numpy dizisine dönüştür
            image_data = base64.b64decode(data['image'].split(',')[1])
            image_buffer = BytesIO(image_data)
            image = Image.open(image_buffer)
            image_array = np.array(image)
            
            # Parametreleri al
            start_x = int(data['startX'])
            start_y = int(data['startY'])
            width = int(data['width'])
            height = int(data['height'])
            aspect_ratio = data['aspectRatio']
            
            # Görüntüyü kırp
            kirpilmis_goruntu = goruntu_kirp(image_array, start_x, start_y, width, height, aspect_ratio)
            
            # Sonuç görüntüsünü base64'e dönüştür
            result_image = Image.fromarray(kirpilmis_goruntu)
            buffer = BytesIO()
            result_image.save(buffer, format='PNG')
            base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return JsonResponse({
                'success': True,
                'image': f'data:image/png;base64,{base64_image}'
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

def goruntu_kirpma(request):
    return render(request, 'core/islemler/goruntu_kirpma.html')

@csrf_exempt
def yakinlastirma_isle(request):
    if request.method == 'POST':
        try:
            # Görüntü kontrolü
            image = request.FILES.get('image')
            if not image:
                return JsonResponse({'error': 'Lütfen bir görüntü yükleyin'}, status=400)
            
            # Dosya boyutu kontrolü (max 10MB)
            if image.size > 10 * 1024 * 1024:
                return JsonResponse({'error': 'Görüntü boyutu 10MB\'dan küçük olmalıdır'}, status=400)

            # Parametreleri al ve doğrula
            try:
                olcek_faktoru = float(request.POST.get('olcek_faktoru', 1.0))
                if not (0.1 <= olcek_faktoru <= 5.0):
                    return JsonResponse({'error': 'Ölçek faktörü 0.1 ile 5.0 arasında olmalıdır'}, status=400)
            except ValueError:
                return JsonResponse({'error': 'Geçersiz ölçek faktörü'}, status=400)

            interpolasyon = request.POST.get('interpolasyon', 'linear')
            if interpolasyon not in ['nearest', 'linear', 'cubic', 'lanczos']:
                return JsonResponse({'error': 'Geçersiz interpolasyon yöntemi'}, status=400)

            anti_aliasing = request.POST.get('antiAliasing', 'true').lower() == 'true'

            # Görüntüyü numpy dizisine çevir
            img_array = read_image_file(image)
            if img_array is None:
                return JsonResponse({'error': 'Görüntü formatı desteklenmiyor'}, status=400)

            # Debug için parametre değerlerini yazdır
            print(f"Ölçekleme parametreleri: faktör={olcek_faktoru}, interpolasyon={interpolasyon}, anti_aliasing={anti_aliasing}")

            # Görüntüyü ölçekle
            yeni_img = goruntu_olcekle(
                image=img_array,
                olcek_faktoru=olcek_faktoru,
                interpolasyon=interpolasyon,
                anti_aliasing=anti_aliasing
            )

            # Belleği temizle
            del img_array

            # Sonuç görüntüsünü base64'e çevir
            processed_image = encode_image_base64(yeni_img)
            del yeni_img

            return JsonResponse({
                'success': True,
                'processed_image': processed_image,
                'parameters': {
                    'olcek_faktoru': olcek_faktoru,
                    'interpolasyon': interpolasyon,
                    'anti_aliasing': anti_aliasing
                }
            })

        except MemoryError:
            return JsonResponse({'error': 'Görüntü işlenirken bellek yetersiz kaldı'}, status=500)
        except Exception as e:
            return JsonResponse({'error': f'İşlem sırasında bir hata oluştu: {str(e)}'}, status=500)

    return JsonResponse({'error': 'Geçersiz istek yöntemi'}, status=405)

def yakinlastirma(request):
    return render(request, 'core/islemler/yakinlastirma.html')

def histogram(request):
    return render(request, 'core/islemler/histogram.html')

@csrf_exempt
def kontrast_isle_view(request):
    """Kontrast ayarlama işlemini gerçekleştiren view fonksiyonu"""
    try:
        if request.method != 'POST':
            return JsonResponse({'error': 'POST isteği gerekli'}, status=400)
            
        # Debug için gelen parametreleri yazdır
        print("Gelen istek parametreleri:")
        print("POST data:", request.POST)
        print("Files:", request.FILES)
            
        # Görüntüyü al
        if 'image' not in request.FILES:
            return JsonResponse({'error': 'Görüntü dosyası gerekli'}, status=400)
            
        # Parametreleri al
        yontem = request.POST.get('yontem', 'dogrusal')
        kontrast_faktor = float(request.POST.get('kontrastFaktor', 1.0))
        gamma = float(request.POST.get('gammaFaktor', 1.0))
        kanal_bazli = request.POST.get('kanalBazli', 'false').lower() == 'true'
        
        print(f"İşlem parametreleri: yontem={yontem}, kontrast_faktor={kontrast_faktor}, gamma={gamma}, kanal_bazli={kanal_bazli}")
        
        # Görüntüyü numpy dizisine dönüştür
        image = read_image_file(request.FILES['image'])
        
        if image is None:
            return JsonResponse({'error': 'Görüntü okunamadı'}, status=400)
            
        print(f"Görüntü boyutu: {image.shape}")
        
        # Kontrast işlemini uygula
        sonuc = kontrast_isle(image, yontem, kontrast_faktor, gamma, kanal_bazli)
        
        print("İşlem tamamlandı, sonuç boyutu:", sonuc.shape)
        
        # Sonucu base64'e dönüştür
        processed_image = encode_image_base64(sonuc)
        
        return JsonResponse({
            'success': True,
            'image': processed_image
        })
        
    except Exception as e:
        print("Hata oluştu:", str(e))
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=400)

def kontrast(request):
    """Kontrast ayarlama sayfasını render eden view"""
    return render(request, 'core/islemler/kontrast.html')

def kenar_bulma(request):
    return render(request, 'core/islemler/kenar_algilama.html')

@csrf_exempt
def kenar_algilama_isle(request):
    """Kenar algılama işlemini gerçekleştiren view fonksiyonu"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST isteği gerekli'}, status=400)
    
    try:
        # Görüntüyü al
        if 'image' not in request.FILES:
            return JsonResponse({'error': 'Görüntü dosyası gerekli'}, status=400)
        
        # Parametreleri al
        yontem = request.POST.get('yontem', 'sobel')
        yon = request.POST.get('yon', 'her_iki')
        alt_esik = int(request.POST.get('altEsik', 100))
        ust_esik = int(request.POST.get('ustEsik', 200))
        on_islem = request.POST.get('onIslemUygula', 'false').lower() == 'true'
        filtre_boyutu = int(request.POST.get('filtreBoyutu', 3))
        
        # Debug için parametreleri yazdır
        print(f"Kenar algılama parametreleri:")
        print(f"Yöntem: {yontem}")
        print(f"Yön: {yon}")
        print(f"Alt eşik: {alt_esik}")
        print(f"Üst eşik: {ust_esik}")
        print(f"Ön işlem: {on_islem}")
        print(f"Filtre boyutu: {filtre_boyutu}")
        
        # Görüntüyü numpy dizisine dönüştür
        image = read_image_file(request.FILES['image'])
        
        if image is None:
            return JsonResponse({'error': 'Görüntü okunamadı'}, status=400)
        
        # Kenar algılama modülünü import et
        from .image_processing.edge_detection import kenar_algilama
        
        # İşlemi uygula
        sonuc = kenar_algilama(
            image=image,
            yontem=yontem,
            yon=yon,
            alt_esik=alt_esik,
            ust_esik=ust_esik,
            on_islem=on_islem,
            filtre_boyutu=filtre_boyutu
        )
        
        # Sonucu base64'e dönüştür
        processed_image = encode_image_base64(sonuc)
        
        return JsonResponse({
            'success': True,
            'image': processed_image
        })
        
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=400)

def gurultu(request):
    return render(request, 'core/islemler/gurultu.html')

def morfolojik(request):
    return render(request, 'core/islemler/morfolojik.html')

def goruntu_dondurme(request):
    return render(request, 'core/islemler/goruntu_dondurme.html')

@csrf_exempt
def histogram_isle_view(request):
    if request.method == 'POST':
        try:
            # Görüntü kontrolü
            image = request.FILES.get('image')
            if not image:
                return JsonResponse({'error': 'Lütfen bir görüntü yükleyin'}, status=400)
            
            # Parametreleri al
            islem_turu = request.POST.get('islemTuru', 'esitleme')
            kanal_bazli = request.POST.get('kanalBazli', 'false').lower() == 'true'
            
            # Germe işlemi için parametreleri al
            min_deger = int(request.POST.get('minDeger', 0))
            max_deger = int(request.POST.get('maxDeger', 255))
            
            # Görüntüyü numpy dizisine çevir
            img_array = read_image_file(image)
            if img_array is None:
                return JsonResponse({'error': 'Görüntü formatı desteklenmiyor'}, status=400)
            
            # Debug için parametre değerlerini yazdır
            print(f"Histogram işlemi parametreleri:")
            print(f"İşlem türü: {islem_turu}")
            print(f"Kanal bazlı: {kanal_bazli}")
            print(f"Min değer: {min_deger}")
            print(f"Max değer: {max_deger}")
            
            # Orijinal görüntünün histogramını hesapla
            orijinal_hist = histogram_hesapla(img_array)
            orijinal_hist_goruntu = histogram_grafigi_olustur(orijinal_hist)
            
            # Histogram işlemini uygula
            islenmiş_goruntu = histogram_isle(
                image=img_array,
                islem_turu=islem_turu,
                min_deger=min_deger,
                max_deger=max_deger,
                kanal_bazli=kanal_bazli
            )
            
            # İşlenmiş görüntünün histogramını hesapla
            islenmiş_hist = histogram_hesapla(islenmiş_goruntu)
            islenmiş_hist_goruntu = histogram_grafigi_olustur(islenmiş_hist)
            
            # Görüntüleri base64'e çevir
            orijinal_hist_base64 = encode_image_base64(orijinal_hist_goruntu)
            islenmiş_hist_base64 = encode_image_base64(islenmiş_hist_goruntu)
            islenmiş_goruntu_base64 = encode_image_base64(islenmiş_goruntu)
            
            return JsonResponse({
                'success': True,
                'processed_image': islenmiş_goruntu_base64,
                'original_histogram': orijinal_hist_base64,
                'processed_histogram': islenmiş_hist_base64
            })
            
        except Exception as e:
            print(f"Hata oluştu: {str(e)}")  # Hatayı sunucu konsoluna yazdır
            return JsonResponse({
                'error': f'İşlem sırasında bir hata oluştu: {str(e)}'
            }, status=500)
    
    return JsonResponse({'error': 'Geçersiz istek yöntemi'}, status=405)

@csrf_exempt
def gurultu_isle(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Sadece POST istekleri kabul edilir.'})
    
    try:
        # Görüntüyü al
        image_file = request.FILES.get('image')
        if not image_file:
            return JsonResponse({'success': False, 'error': 'Görüntü bulunamadı.'})
        
        # Parametreleri al
        islem_turu = request.POST.get('islem_turu')
        
        # Görüntüyü numpy dizisine çevir
        img = Image.open(image_file)
        if img.mode in ['RGBA', 'LA']:
            # RGBA veya LA (gri+alfa) görüntüyü RGB'ye dönüştür
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])  # Alfa kanalını kullanarak birleştir
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        image_array = np.array(img)
        
        # Gürültü işleme modülünü import et
        from .image_processing.noise import gurultu_isle
        
        if islem_turu == 'ekle':
            gurultu_turu = request.POST.get('gurultu_turu')
            yogunluk = float(request.POST.get('yogunluk', '10'))
            
            # Gürültü ekleme işlemi
            processed_image = gurultu_isle(
                image=image_array,
                islem_turu='ekle',
                gurultu_turu=gurultu_turu,
                yogunluk=yogunluk
            )
        
        else:  # islem_turu == 'temizle'
            filtre_turu = request.POST.get('filtre_turu')
            filtre_boyutu = int(request.POST.get('filtre_boyutu', '3'))
            
            # Gürültü temizleme işlemi
            processed_image = gurultu_isle(
                image=image_array,
                islem_turu='temizle',
                filtre_turu=filtre_turu,
                filtre_boyutu=filtre_boyutu
            )
        
        # İşlenmiş görüntüyü base64'e çevir
        img = Image.fromarray(processed_image.astype('uint8'))
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return JsonResponse({
            'success': True,
            'processed_image': img_str
        })
        
    except Exception as e:
        print(f"Hata: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        })

@csrf_exempt
def morfolojik_isle_view(request):
    """Morfolojik işlemleri gerçekleştiren view fonksiyonu"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST isteği gerekli'}, status=400)
    
    try:
        # Görüntüyü al
        if 'image' not in request.FILES:
            return JsonResponse({'error': 'Görüntü dosyası gerekli'}, status=400)
        
        # Parametreleri al
        islem_turu = request.POST.get('islemTuru')
        yapisal_element = request.POST.get('yapisalElement')
        genislik = int(request.POST.get('genislik', 3))
        yukseklik = int(request.POST.get('yukseklik', 3))
        on_islem = request.POST.get('onIslemUygula', 'false').lower() == 'true'
        
        # Debug için parametreleri yazdır
        print(f"Morfolojik işlem parametreleri:")
        print(f"İşlem türü: {islem_turu}")
        print(f"Yapısal element: {yapisal_element}")
        print(f"Boyutlar: {genislik}x{yukseklik}")
        print(f"Ön işlem: {on_islem}")
        
        # Görüntüyü numpy dizisine dönüştür
        image = read_image_file(request.FILES['image'])
        
        if image is None:
            return JsonResponse({'error': 'Görüntü okunamadı'}, status=400)
        
        # Morfolojik işlem modülünü import et
        from .image_processing.morphology import morfolojik_isle
        
        # İşlemi uygula
        sonuc = morfolojik_isle(
            image=image,
            islem_turu=islem_turu,
            yapisal_element_sekli=yapisal_element,
            genislik=genislik,
            yukseklik=yukseklik,
            on_islem=on_islem
        )
        
        # Sonucu base64'e dönüştür
        processed_image = encode_image_base64(sonuc)
        
        return JsonResponse({
            'success': True,
            'image': processed_image
        })
        
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=400)
