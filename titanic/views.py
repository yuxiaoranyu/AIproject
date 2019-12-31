from django.shortcuts import render
from .ml import MachineLearn
from .ml_new import ML
from django.http import JsonResponse

# Create your views her

def index(request):
    return render(request,'titanic/index.html')

def index2(request):
    return render(request,'titanic/index2.html')

def pred(request):
    sex=request.POST.get('sex',None)
    alone=float(request.POST.get('alone',None))
    age = float(request.POST.get('age', None))
    fare = float(request.POST.get('fare', None))
    logic_select = request.POST.get('logic_select', None)
    ml=MachineLearn(sex,age,fare,alone)
    res_pred=''
    if logic_select=='KNN':
        res_pred=ml.KNN()
    elif logic_select=='LogicRegression':
        res_pred=ml.LogsticRegression()
    elif logic_select=='DecisionTree':
        res_pred=ml.DecisionTree()
    elif logic_select=='RandomForest':
        res_pred=ml.RandomForest()
    elif logic_select=='SVM':
        res_pred=ml.SVM()
    elif logic_select=='KMeans':
        res_pred=ml.KMeans()
    elif logic_select=='Bagging':
        res_pred=ml.Bagging()
    elif logic_select=='Adaboost':
        res_pred=ml.Adaboost()
    return JsonResponse({"msg": res_pred[0], "acc": res_pred[1]})



def pred2(request):
    sex=float(request.POST.get('sex',None))
    initial = float(request.POST.get('initial', None))
    age = float(request.POST.get('age', None))
    sibsp = float(request.POST.get('sibsp', None))
    parch = float(request.POST.get('parch', None))
    fare = float(request.POST.get('fare', None))
    embarked = float(request.POST.get('embarked', None))
    pclass = float(request.POST.get('pclass', None))
    logic_select = float(request.POST.get('logic_select', None))

    ml=ML(sex,initial,age,sibsp,parch,fare,embarked,pclass)
    res_pred = ''
    if logic_select == 'KNN':
        res_pred = ml.KNN()
    elif logic_select == 'LogicRegression':
        res_pred = ml.LogsticRegression()
    elif logic_select == 'DecisionTree':
        res_pred = ml.DecisionTree()
    elif logic_select == 'RandomForest':
        res_pred = ml.RandomForest()
    elif logic_select == 'SVM':
        res_pred = ml.SVM()
    elif logic_select == 'KMeans':
        res_pred = ml.KMeans()
    elif logic_select == 'Bagging':
        res_pred = ml.Bagging()
    elif logic_select == 'Adaboost':
        res_pred = ml.Adaboost()
    return JsonResponse({'msg': res_pred[0], 'acc': res_pred[1]})




