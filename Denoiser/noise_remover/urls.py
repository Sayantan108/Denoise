from django.urls import path
from . import views

urlpatterns = [
    path('',views.post_noisy_image,name='denoise'),
    path('success/', views.uploadok, name = 'success'),
]

