from django.urls import path
from . import views

app_name = 'core'

urlpatterns = [
    path('', views.anasayfa, name='anasayfa'),
    path('gri-donusum/', views.gri_donusum, name='gri_donusum'),
    path('binary-donusum/', views.binary_donusum, name='binary_donusum'),
    path('goruntu-dondurme/', views.goruntu_dondurme, name='goruntu_dondurme'),
    path('goruntu-dondurme/isle/', views.goruntu_dondurme_isle, name='goruntu_dondurme_isle'),
    path('goruntu-kirpma/', views.goruntu_kirpma, name='goruntu_kirpma'),
    path('goruntu-kirpma/isle/', views.goruntu_kirpma_isle, name='goruntu_kirpma_isle'),
    path('yakinlastirma/', views.yakinlastirma, name='yakinlastirma'),
    path('yakinlastirma/isle/', views.yakinlastirma_isle, name='yakinlastirma_isle'),
    path('histogram/', views.histogram, name='histogram'),
    path('histogram/isle/', views.histogram_isle_view, name='histogram_isle'),
    path('kontrast/', views.kontrast, name='kontrast'),
    path('kontrast/isle/', views.kontrast_isle_view, name='kontrast_isle'),
    path('kenar-bulma/', views.kenar_bulma, name='kenar_bulma'),
    path('gurultu/', views.gurultu, name='gurultu'),
    path('morfolojik/', views.morfolojik, name='morfolojik'),
] 