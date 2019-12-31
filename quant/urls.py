from django.urls import path
from .views import *

urlpatterns = [

    path("setcode/",setcode,name="setcode"),
    path("catch/",catch),
    path("outcsv/",outcsv),
    path("dodata/",dodata),
    path("pred/",pred),
    path("back/",back),
    path('',index,name="index"),
]