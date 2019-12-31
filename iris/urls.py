from django.urls import path
from .views import *

urlpatterns=[
    path('linear_pred/',linear_pred,name='linear_pred'),
    path('pred/',pred,name='pred'),
    path('',index,name='index'),
]