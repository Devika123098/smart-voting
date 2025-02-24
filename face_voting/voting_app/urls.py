from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),  # Home Page
    path('register/', views.register, name='register'),  # Registration Page
    path('vote/', views.vote, name='vote'),  # Voting Page
]