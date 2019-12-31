from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn import pipeline
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.utils import shuffle
import sklearn.metrics as sm
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
import neurolab as nl
import warnings
warnings.filterwarnings('ignore')


class MachineLearn(object):
    def __init__(self, title, positionInfo, area, beds, rooms, toward, renovation):
        self.title = title
        self.positionInfo = int(positionInfo)
        self.area = int(area)
        self.beds = int(beds)
        self.rooms = int(rooms)
        self.toward = toward
        self.renovation = int(renovation)
        # 1、构造数据
        # todo：构造训练数据
        PATH = os.path.abspath(os.path.dirname(__file__)) + os.sep + 'data' + os.sep + 'preprocess_gz_house.csv'
        data_train = pd.read_csv(PATH)
        data_train = data_train.iloc[:, 1:]
        # print(data_train.head())
        # todo：清洗特征集
        data_x = data_train[
            ['gz_title', 'gz_area', 'gz_beds', 'gz_rooms', 'gz_toward', 'gz_renovation', 'gz_positionInfo']]
        # todo：毛坯房的处理
        data_x.loc[data_x['gz_renovation'] == '毛坯', 'gz_renovation'] = '-1'
        data_x.loc[data_x['gz_renovation'] == '其他', 'gz_renovation'] = '0'
        data_x['gz_renovation'] = data_x['gz_renovation'].astype('float')
        # print(data_x['gz_renovation'])
        # todo：gz_title标签编码化
        data_x['gz_title'] = data_x['gz_title'].replace(
            ['tianhe', 'yuexiu', 'liwan', 'haizhu', 'panyu', 'baiyun', 'huangpu', 'conghua', 'zengcheng', 'huadu',
             'nansha', 'nanhai', 'shunde'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        if self.title == 'tianhe':
            self.title = 0
        elif self.title == 'yuexiu':
            self.title = 1
        elif self.title == 'liwan':
            self.title = 2
        elif self.title == 'haizhu':
            self.title = 3
        elif self.title == 'panyu':
            self.title = 4
        elif self.title == 'baiyun':
            self.title = 5
        elif self.title == 'huangpu':
            self.title = 6
        elif self.title == 'conghua':
            self.title = 7
        elif self.title == 'zengcheng':
            self.title = 8
        elif self.title == 'huadu':
            self.title = 9
        elif self.title == 'nansha':
            self.title = 10
        elif self.title == 'nanhai':
            self.title = 11
        elif self.title == 'shunde':
            self.title = 12

        # todo：area离散化
        # print(data_x['gz_area'].describe())
        # mean 87.985115
        # std 35.337333
        # min 15.100000
        # 25 % 65.000000
        # 50 % 83.000000
        # 75 % 103.630000
        # max 438.650000
        data_x.loc[data_x['gz_area'] <= 65, 'area_new'] = 0
        data_x.loc[(data_x['gz_area'] > 65) & (data_x['gz_area'] <= 83), 'area_new'] = 1
        data_x.loc[(data_x['gz_area'] > 83) & (data_x['gz_area'] <= 103), 'area_new'] = 2
        data_x.loc[(data_x['gz_area'] > 103) & (data_x['gz_area'] <= 270), 'area_new'] = 3
        data_x.loc[data_x['gz_area'] > 270, 'area_new'] = 4
        data_x['gz_area'] = data_x['area_new']
        if self.area > 270:
            self.area = 4
        elif self.area > 103:
            self.area = 3
        elif self.area > 83:
            self.area = 2
        elif self.area > 65:
            self.area = 1
        else:
            self.area = 0

        # todo:gz_toward标签编码化
        data_x['gz_toward'] = data_x['gz_toward'].replace(['北', '西', '东', '南'], [0, 1, 2, 3])
        if self.toward == 'north':
            self.toward = 0
        elif self.toward == 'west':
            self.toward = 1
        elif self.toward == 'east':
            self.toward = 2
        elif self.toward == 'south':
            self.toward = 3
        # print(data_x['gz_toward'])

        # todo:gz_positionInfo离散化
        # print(data_x['gz_positionInfo'].describe())
        # count 3609.000000
        # mean 16.214741
        # std 9.806761
        # min 1.000000
        # 25 % 8.000000
        # 50 % 13.000000
        # 75 % 24.000000
        # max 54.000000
        data_x.loc[data_x['gz_positionInfo'] <= 8, 'positionInfo_new'] = 0
        data_x.loc[(data_x['gz_positionInfo'] > 8) & (data_x['gz_positionInfo'] <= 13), 'positionInfo_new'] = 1
        data_x.loc[(data_x['gz_positionInfo'] > 13) & (data_x['gz_positionInfo'] <= 24), 'positionInfo_new'] = 2
        data_x.loc[(data_x['gz_positionInfo'] > 24) & (data_x['gz_positionInfo'] <= 39), 'positionInfo_new'] = 3
        data_x.loc[data_x['gz_positionInfo'] > 39, 'positionInfo_new'] = 4
        data_x['gz_positionInfo'] = data_x['positionInfo_new']

        if self.positionInfo > 39:
            self.positionInfo = 4
        elif self.positionInfo > 24:
            self.positionInfo = 3
        elif self.positionInfo > 13:
            self.positionInfo = 2
        elif self.positionInfo > 8:
            self.positionInfo = 1
        else:
            self.positionInfo = 0

        # 删除不需要的列
        data_x.drop(['positionInfo_new', 'area_new'], axis=1, inplace=True)
        # 构造标签集
        # 总价预测
        # print(data_train.head())
        Total_Y = data_train['gz_totalPrice']
        Unit_Y = data_train['gz_unitPrice']
        Y = pd.concat([Total_Y, Unit_Y], axis=1)
        # 先打乱
        X, Y = shuffle(data_x, Y, random_state=8)
        # 训练集和测试集分割
        self.x_train, self.x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=8)
        self.total_y_train = y_train['gz_totalPrice']
        self.unit_y_train = y_train['gz_unitPrice']
        self.total_y_test = y_test['gz_totalPrice']
        self.unit_y_test = y_test['gz_unitPrice']

        # todo：构造测试数据
        self.pred = [[self.title, self.area, self.beds, self.rooms, self.toward, self.renovation, self.positionInfo]]

    def LineRegressor(self):
        # 线性回归模型
        # 1、构造模型
        model_total = LinearRegression()
        model_unit = LinearRegression()
        # 2、模型训练
        model_total.fit(self.x_train, self.total_y_train)
        model_unit.fit(self.x_train, self.unit_y_train)
        # 3、模型预测
        total_pred_y = int(model_total.predict(self.pred))
        total_pred_y = 'Line总价：' + str(total_pred_y) + '万元'
        unit_pred_y = int(model_unit.predict(self.pred))
        unit_pred_y = 'Line单价：' + str(unit_pred_y) + '元/㎡'
        # 4、模型评估
        total_y = model_total.predict(self.x_test)
        unit_y = model_unit.predict(self.x_test)
        total_acc = sm.r2_score(self.total_y_test, total_y)
        total_acc = '模型R2评分：' + str(round(total_acc, 2))
        unit_acc = sm.r2_score(self.unit_y_test, unit_y)
        unit_acc = '模型R2评分：' + str(round(unit_acc, 2))
        return [total_pred_y, total_acc], [unit_pred_y, unit_acc]

    def PolyRegressor(self):
        #多项式回归
        #1、构造模型
        model_total=pipeline.make_pipeline(PolynomialFeatures(3),LinearRegression())
        model_unit=pipeline.make_pipeline(PolynomialFeatures(3),LinearRegression())
        #2、模型训练
        model_total.fit(self.x_train,self.total_y_train)
        model_unit.fit(self.x_train,self.unit_y_train)
        #3、模型预测
        total_pred_y=int(model_total.predict(self.pred))
        total_pred_y='Poly总价：'+str(total_pred_y)+'万元'
        unit_pred_y=int(model_unit.predict(self.pred))
        unit_pred_y='Poly单价：'+str(unit_pred_y)+'元/㎡'
        #4、模型评估
        total_y=model_total.predict(self.x_test)
        unit_y=model_unit.predict(self.x_test)
        total_acc=sm.r2_score(self.total_y_test,total_y)
        total_acc='模型R2评分：'+str(round(total_acc,2))
        unit_acc = sm.r2_score(self.unit_y_test, unit_y)
        unit_acc = '模型R2评分：' + str(round(unit_acc,2))
        return [total_pred_y, total_acc], [unit_pred_y, unit_acc]

    def DecisionTree(self):
        #决策树模型
        # 1、构造模型
        model_total = DecisionTreeRegressor(max_depth=6)
        model_unit = DecisionTreeRegressor(max_depth=6)
        # 2、模型训练
        model_total.fit(self.x_train, self.total_y_train)
        model_unit.fit(self.x_train, self.unit_y_train)
        # 3、模型预测
        total_pred_y = int(model_total.predict(self.pred))
        total_pred_y = 'Deci总价：' + str(total_pred_y) + '万元'
        unit_pred_y = int(model_unit.predict(self.pred))
        unit_pred_y = 'Deci单价：' + str(unit_pred_y) + '元/㎡'
        # 4、模型评估
        total_y = model_total.predict(self.x_test)
        unit_y = model_unit.predict(self.x_test)
        total_acc = sm.r2_score(self.total_y_test, total_y)
        total_acc = '模型R2评分：' + str(round(total_acc, 2))
        unit_acc = sm.r2_score(self.unit_y_test, unit_y)
        unit_acc = '模型R2评分：' + str(round(unit_acc, 2))
        return [total_pred_y, total_acc], [unit_pred_y, unit_acc]

    def SVR(self):
        #支持向量机模型
        #1、构造最优模型
        model_total = SVR(kernel='poly',C=2.0,degree=4)
        model_unit = SVR(kernel='poly',C=2.0,degree=4)
        # 2、模型训练
        model_total.fit(self.x_train, self.total_y_train)
        model_unit.fit(self.x_train, self.unit_y_train)
        # 3、模型预测
        total_pred_y = int(model_total.predict(self.pred))
        total_pred_y = 'SVR总价：' + str(total_pred_y) + '万元'
        unit_pred_y = int(model_unit.predict(self.pred))
        unit_pred_y = 'SVR单价：' + str(unit_pred_y) + '元/㎡'
        # 4、模型评估
        total_y = model_total.predict(self.x_test)
        unit_y = model_unit.predict(self.x_test)
        total_acc = sm.r2_score(self.total_y_test, total_y)
        total_acc = '模型R2评分：' + str(round(total_acc, 2))
        unit_acc = sm.r2_score(self.unit_y_test, unit_y)
        unit_acc = '模型R2评分：' + str(round(unit_acc, 2))
        return [total_pred_y, total_acc], [unit_pred_y, unit_acc]

    def Bagging(self):
        model_total = BaggingRegressor(KNeighborsRegressor(n_neighbors=3),n_estimators=700)
        model_unit = BaggingRegressor(KNeighborsRegressor(n_neighbors=3),n_estimators=700)
        # 2、模型训练
        model_total.fit(self.x_train, self.total_y_train)
        model_unit.fit(self.x_train, self.unit_y_train)
        # 3、模型预测
        total_pred_y = int(model_total.predict(self.pred))
        total_pred_y = 'Bag总价：' + str(total_pred_y) + '万元'
        unit_pred_y = int(model_unit.predict(self.pred))
        unit_pred_y = 'Bag单价：' + str(unit_pred_y) + '元/㎡'
        # 4、模型评估
        total_y = model_total.predict(self.x_test)
        unit_y = model_unit.predict(self.x_test)
        total_acc = sm.r2_score(self.total_y_test, total_y)
        total_acc = '模型R2评分：' + str(round(total_acc, 2))
        unit_acc = sm.r2_score(self.unit_y_test, unit_y)
        unit_acc = '模型R2评分：' + str(round(unit_acc, 2))
        return [total_pred_y, total_acc], [unit_pred_y, unit_acc]

    def DNN(self):
        #构造值域
        nn_minmax=[]
        for i in range(self.x_train.shape[-1]):
            min_x=self.x_train.iloc[:,i].min()
            max_x=self.x_train.iloc[:,i].max()
            nn_minmax.append([min_x,max_x])
        model=nl.net.newff(nn_minmax,[28,28,28,1])
        model.trainf=nl.train.train_gd#设置为梯度下降
        #模型训练
        x_data=self.x_train
        y_data=self.total_y_train.values.reshape(-1,1)
        error=model.train(x_data,y_data,epochs=2,show=1,lr=0.01)
        #模型预测
        pred=model.sim(self.pred)[0][0]
        #模型评估
        pred_test_y=model.sim(self.x_test)
        metr=round(sm.r2_score(pred_test_y,self.total_y_test),2)
        return [pred,metr],[pred,metr]

# c = MachineLearn("tianhe", "20", '150', '2', '2', "east", "1")
# print(c.LineRegressor())
# print(c.PolyRegressor())
# print(c.DecisionTree())
# print(c.SVR())
# print(c.Bagging())
# print(c.DNN())