from django.shortcuts import render
from .quant import Quant
from django.http import JsonResponse


# Create your views here.
def index(request):
    return render(request, 'quant/index.html')


def setcode(request):
    code = request.POST.get('code', None)
    quant = Quant()
    stock_list = quant.get_realtime_stock_info(code)
    msg = "<table width=490px><tr><td>代码</td><td>名称</td><td>开盘价</td><td>上一天收盘价</td><td>价格</td><td>获取数据</td></tr>"
    for i in range(len(stock_list)):
        rowdict = stock_list[i]
        button_str = "<input type=button value='提取' id=begin" + str(i) + " onclick='quantAction(" + str(i) + ")'>"
        msg += "<tr><td>" + rowdict["code"] + "</td><td>" + rowdict["name"] + "</td><td>" + rowdict[
            "open"] + "</td><td>" + rowdict["pre_close"] + "</td><td>" + rowdict[
                   "price"] + "</td><td>" + button_str + "</td>" + "</tr>"
    msg += "</table>"
    return JsonResponse({'msg': msg})


def catch(request):
    catch_code = request.POST.get("catch_code",None)
    start_time = request.POST.get("start_time",None)
    end_time = request.POST.get("end_time",None)
    quant = Quant()
    quant.catch_data(catch_code,start_time,end_time)
    return JsonResponse({"msg":f"成功获取{catch_code}数据！"})

def outcsv(request):
    catch_code = request.POST.get("catch_code", None)
    quant = Quant()
    stock_path = quant.out_csv(catch_code)
    return JsonResponse({"msg": f"数据保存路径：{stock_path}"})

def dodata(request):
    catch_code = request.POST.get('catch_code', None)
    quant = Quant()
    train_len, test_len = quant.dodata(catch_code)  # 返回训练集和测试集
    return JsonResponse({'msg': f'数据处理成功-->训练数据：{str(train_len)}条；测试数据：{str(test_len)}'})

def pred(request):
    catch_code = request.POST.get('catch_code', None)
    logic_select = request.POST.get('logic_select', None)
    quant = Quant()
    res_pred = ''
    if logic_select == 'LinearRegression':
        res_pred = quant.LinearRegession(catch_code)
    elif logic_select == 'SVR':
        res_pred = quant.SVR(catch_code)
    return JsonResponse({'msg': res_pred})


def back(request):
    quant_select = request.POST.get('quant_select', None)
    quant = Quant()
    res_pred = ''
    if quant_select == 'daily':
        res_pred = quant.daily_stats()
    elif quant_select == 'id_':
        res_pred = quant.id_stats()
    elif quant_select == 'overnight':
        res_pred = quant.overnight_stats()
    elif quant_select == 'custom_':
        res_pred = quant.custom_stats()
    return JsonResponse({'msg': res_pred})
