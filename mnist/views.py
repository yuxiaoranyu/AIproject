from django.shortcuts import render
import os
from django.http import JsonResponse
from .ml import MachineLearn

# Create your views here.
def index(request):
    return render(request,'mnist/index.html')

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
    logic_select=request.POST.get('logic_select',None)
    res_pred=''
    ml=MachineLearn(file_path)
    if logic_select=='DNN_Keras':
        res_pred=ml.DNN_keras()
    elif logic_select=='MLP_Keras':
        res_pred=ml.MLP_keras()
    elif logic_select=='DNN_Tensorflow':
        res_pred=ml.DNN_Tensorflow()
    return JsonResponse({'msg':res_pred[0],'acc':res_pred[1]})