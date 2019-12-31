from django.shortcuts import render
import os
from django.http import JsonResponse
from .catdog import CatDog
# Create your views here.
def index(request):
    return render(request,'catdog/index.html')

def upload(request):
    if request.method=='POST':
        ret={'status':False,'data':False,'error':None}
        try:
            img=request.FILES.get('img')
            FILE_PATH=os.path.abspath(os.path.dirname(__file__))+os.sep+'static'+os.sep+img.name
            OUT_PATH=FILE_PATH
            FILE_PATH_URL='/static/'+img.name
            f=open(FILE_PATH,'wb')
            for chunk in img.chunks(chunk_size=10241024):
                f.write(chunk)
            ret['status']=True
            ret['status']=FILE_PATH_URL
        except Exception as e:
            ret['error']=e
            return JsonResponse({'file_path':'','file_path_url':'','status':ret['status'],'error':ret['error']})
        finally:
            f.close()
        return JsonResponse({'file_path':OUT_PATH,'file_path_url':FILE_PATH_URL,'status':ret['status'],'error':ret['error']})


def pred(request):
    file_path=request.POST.get('file_path',None)
    model=CatDog(file_path)
    model.train_cat_dog()
    res=model.pred_cat_dog()
    return JsonResponse({'msg':res})