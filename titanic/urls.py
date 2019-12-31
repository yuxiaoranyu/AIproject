from django.urls import path
from .views import *

urlpatterns=[
    path('pred/',pred,name='pred'),
    path('pred2/',pred,name='pred2'),
    path('index2/',index2,name='index2'),
    path('',index,name='index'),
]