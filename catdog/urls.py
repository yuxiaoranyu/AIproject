from django.urls import path
from .views import *

urlpatterns = [

    path("pred/",pred,name="pred"),
    path('upload/',upload,name='upload'),
    path('',index,name="index"),
]