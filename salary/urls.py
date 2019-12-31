from django.urls import path
from .views import *

urlpatterns = [
    # path("catch/",catch),
    # path("outcsv/",outcsv),
    # path("dodata/",do_data),
    # path("pred/",pred),
    path('',index,name="index"),
]