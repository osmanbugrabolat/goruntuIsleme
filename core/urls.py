from django.urls import path
from . import views

app_name = 'core'

urlpatterns = [
    path('', views.anasayfa, name='anasayfa'),
    path('gri-donusum/', views.gri_donusum, name='gri_donusum'),
    path('binary-donusum/', views.binary_donusum, name='binary_donusum'),
    path('goruntu-dondurme/', views.goruntu_dondurme, name='goruntu_dondurme'),
    path('goruntu-kirpma/', views.goruntu_kirpma, name='goruntu_kirpma'),
    path('yakinlastirma/', views.yakinlastirma, name='yakinlastirma'),
    path('histogram/', views.histogram, name='histogram'),
    path('kontrast/', views.kontrast, name='kontrast'),
    path('kenar-bulma/', views.kenar_bulma, name='kenar_bulma'),
    path('gurultu/', views.gurultu, name='gurultu'),
    path('morfolojik/', views.morfolojik, name='morfolojik'),
] 