from django.shortcuts import render
from .ml import MachineLearn
from .lianjia import House
from django.http import JsonResponse
# Create your views here.
def index(request):
    return render(request,'rent/index.html')

def catch(request):
    house=House()
    house.get_house()
    total_count=house.get_count()
    return JsonResponse({'msg':total_count})

def outcsv(request):
    house=House()
    result=house.out_csv()
    return JsonResponse(result)

def do_data(request):
    house=House()
    result=house.do_data()
    return JsonResponse({'msg':result})

def pred(requset):
    positionInfo=requset.POST.get('positionInfo',None)
    renovation=requset.POST.get('renovation',None)
    area=requset.POST.get('area',None)
    beds = requset.POST.get('beds', None)
    rooms = requset.POST.get('rooms', None)
    title = requset.POST.get('title', None)
    toward = requset.POST.get('toward', None)
    price_select = requset.POST.get('price_select', None)
    logic_select = requset.POST.get('logic_select', None)
    ml=MachineLearn(title,positionInfo,area,beds,rooms,toward,renovation)
    re_pred=''
    if logic_select=='LinearRegression':
        if price_select=='total':
            re_pred=ml.LineRegressor()[0]
        else:
            re_pred=ml.LineRegressor()[1]
    elif logic_select=='PolyRegression':
        if price_select=='total':
            re_pred=ml.PolyRegressor()[0]
        else:
            re_pred=ml.PolyRegressor()[1]
    elif logic_select=='DecisionTree':
        if price_select=='total':
            re_pred=ml.DecisionTree()[0]
        else:
            re_pred=ml.DecisionTree()[1]
    elif logic_select=='SVR':
        if price_select=='total':
            re_pred=ml.SVR()[0]
        else:
            re_pred=ml.SVR()[1]
    elif logic_select=='Bagging':
        if price_select=='total':
            re_pred=ml.Bagging()[0]
        else:
            re_pred=ml.Bagging()[1]

    return JsonResponse({'msg':re_pred[0],'acc':re_pred[1]})