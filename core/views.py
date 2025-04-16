from django.shortcuts import render

# Create your views here.

def anasayfa(request):
    return render(request, 'core/anasayfa.html')

def gri_donusum(request):
    return render(request, 'core/islemler/gri_donusum.html')

def binary_donusum(request):
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
