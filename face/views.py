from django.shortcuts import render
import os
from django.http import JsonResponse
from .pgm import Pgm
from .ml import MachineLearn
from .dl import Deeplearn

# Create your views here.
def index(request):
    return render(request,'face/index.html')

def upload(request):
    if request.method=='POST':
        ret={'status':False,'data':None,'error':None}
        try:
            img=request.FILES.get('img')
            FILE_PATH=os.path.abspath(os.path.dirname(__file__))+os.sep+'static'+os.sep+img.name
            OUT_PATH=FILE_PATH+'.jpg'
            FILE_PATH_URL='/static/'+img.name+'.jpg'
            f=open(FILE_PATH,'wb')
            for chunk in img.chunks(chunk_size=10241024):
                f.write(chunk)
            ret['static']=True
            #将pgm的图片保存为jpg格式
            pgm=Pgm()
            pgm.pgm_to_jpg(FILE_PATH,OUT_PATH)
            ret['data']=os.path.join('static',img.name)
        except Exception as e:
            ret['error']=e
            print(f'发生了异常：{e}')
            return JsonResponse({"file_path":"","file_path_url":"","status":ret["status"],"error":ret["error"]})
        finally:
            f.close()
        return JsonResponse({"file_path":FILE_PATH,"file_path_url":FILE_PATH_URL,"status":ret["status"],"error":ret["error"]})

def pred(request):
    file_path=request.POST.get('file_path',None)
    pca=int(request.POST.get('pca',None))
    logic_select=request.POST.get('logic_select',None)
    res_pred=""
    ml=MachineLearn(file_path,pca_k=pca)
    if logic_select=="KNN":
        res_pred=ml.KNN()
    elif logic_select=='LogicRegression':
        res_pred=ml.LogsticRegressor()
    elif logic_select=='DecisionTree':
        res_pred=ml.DecisionTree()
    elif logic_select=='RandomForest':
        res_pred=ml.RandomFroest()
    elif logic_select=='SVM':
        res_pred=ml.SVM()
    elif logic_select=='ENSEMBLE':
        res_pred=ml.Bagging()
    elif logic_select=='DNN':
        dl=Deeplearn(file_path,pca)
        res_pred=dl.Keras_DNN()

    return JsonResponse({'msg':res_pred[0],'acc':res_pred[1]})