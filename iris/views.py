from django.shortcuts import render
from .ml import MachineLearn
from django.http import JsonResponse

# Create your views here.

def index(request):
    return render(request,"iris/index.html")

def linear_pred(request):
    petal_width = float(request.POST.get("petal_width",None))
    linear_select = request.POST.get("linear_select",None)
    ml = MachineLearn()
    if linear_select == "LinearRegression":
        res_pred = ml.LineRegression(petal_width)
        return JsonResponse({"msg":res_pred})
    elif linear_select == "PolyRegression":
        res_pred = ml.PolyRegression(petal_width)
        return JsonResponse({"msg":res_pred})


def pred(request):
    petal_width2 = float(request.POST.get("petal_width2",None))
    petal_length = float(request.POST.get("petal_length",None))
    sepal_width = float(request.POST.get("sepal_width",None))
    sepal_length = float(request.POST.get("sepal_length",None))
    logic_select = request.POST.get("logic_select",None)
    pred = [[sepal_length,sepal_width,petal_length,petal_width2]]
    ml = MachineLearn()
    res_pred = ""
    if logic_select == "KNN":
        res_pred = ml.KNN(k_max=5,pred=pred)
    elif logic_select == "LogicRegression":
        res_pred = ml.LogsticRegression(pred=pred)
    elif logic_select == "DecisionTree":
        res_pred = ml.DecisonTree(pred=pred)
    elif logic_select == "RandomForest":
        res_pred = ml.RandomForest(pred=pred)
    elif logic_select == "SVM":
        res_pred = ml.SVM(pred=pred)
    elif logic_select == "KMeans":
        res_pred = ml.Kmeans(pred=pred)

    return JsonResponse({"msg":res_pred[0],"acc":res_pred[1]})




