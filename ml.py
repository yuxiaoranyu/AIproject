from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn import pipeline
import os
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
import numpy as np
import pandas as pd
import sklearn.preprocessing as sp
from sklearn.utils import shuffle
import sklearn.metrics as sm
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingRegressor
import neurolab as nl
import warnings

warnings.filterwarnings('ignore')


class MachineLearn(object):
    def __init__(self,city, edu, year):
        self.city = city
        self.edu = edu
        self.year = int(year)
        # 1、构造数据
        # todo：构造训练数据
        PATH = os.path.abspath(os.path.dirname(__file__)) + os.sep + 'data' + os.sep + 'pre_python.csv'
        data_train = pd.read_csv(PATH)
        data_train = data_train.dropna(subset=['pre_salary', 'pre_edu', 'pre_worktime'], how='any')
        data_train = data_train.iloc[:, 1:]
        # print(data_train.head())
        # todo：清洗特征集
        data_x = data_train[
            ['pre_title', 'pre_salary', 'pre_worktime', 'pre_city', 'pre_edu']]
        #todo:城市标签编码化
        data_x['pre_city'] = data_x['pre_city'].replace(['北上广深', '新兴城市', '发展城市', '尚未兴起'], [0, 1, 2, 3])
        if self.city == '北京' or self.city == '上海' or self.city == '广州' or self.city == '深圳':
            self.city=0
        elif self.city == '南京' or self.city == '杭州' or self.city == '成都' \
                or self.city == '苏州' or self.city == '武汉' or self.city == '西安' \
                or self.city == '长沙' or self.city == '重庆' or self.city == '昆山':
            self.city=1
        elif self.city == '佛山' or self.city == '合肥' or self.city == '珠海' or self.city == '无锡' \
                or self.city == '东莞' or self.city == '福州' or self.city == '天津' or self.city == '郑州' \
                or self.city == '厦门' or self.city == '青岛' or self.city == '济南':
            self.city=2
        else:
            self.city=3
        # todo:学历标签编码化
        data_x['pre_edu'] = data_x['pre_edu'].replace(['大专', '本科', '硕士', '博士'], [0, 1, 2, 3])
        if self.edu == '大专':
            self.edu = 0
        elif self.edu == '本科':
            self.edu = 1
        elif self.edu == '硕士':
            self.edu = 2
        elif self.edu == '博士':
            self.edu = 3
        # print(data_x['pre_edu'])
        # todo：worktime离散化
        # print(data_x['pre_worktime'].describe())
        data_x.loc[data_x['pre_worktime'] == 0, 'worktime_new'] = 0
        data_x.loc[(data_x['pre_worktime'] >= 1) & (data_x['pre_worktime'] < 3), 'worktime_new'] = 1
        data_x.loc[(data_x['pre_worktime'] >= 3) & (data_x['pre_worktime'] < 5), 'worktime_new'] = 2
        data_x.loc[data_x['pre_worktime'] >= 5, 'worktime_new'] = 3
        data_x['pre_worktime'] = data_x['worktime_new']
        if self.year > 5:
            self.year = 3
        elif self.year > 3:
            self.year = 2
        elif self.year > 1:
            self.year = 1
        else:
            self.year = 0
        data_x.drop(['pre_title','worktime_new','pre_edu'],axis=1,inplace=True)
        Salary_Y = data_train['pre_salary']
        Y = pd.concat([Salary_Y], axis=1)
        # 先打乱
        X, Y = shuffle(data_x, Y, random_state=20)
        # print(X.shape,Y.shape)
        # print('=============')
        # 训练集和测试集分割
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.25, random_state=10)
        self.salary_y_train = self.y_train['pre_salary']
        self.salary_y_test = self.y_test['pre_salary']
        # print(self.x_train.shape)
        # print(self.x_test.shape)
        # print(self.y_train.shape)
        # print(self.y_test.shape)
        # todo：构造测试数据
        self.pred = [[self.city, self.edu, self.year]]
        print(self.pred)

    def LineRegressor(self):
        # 线性回归模型
        # 1、构造模型
        model_total = LinearRegression()
        # 2、模型训练
        model_total.fit(self.x_train, self.salary_y_train)
        # 3、模型预测
        salary_pred_y = int(model_total.predict(self.pred))
        salary_pred_y = 'Line总价：' + str(salary_pred_y) + 'K/月'
        # 4、模型评估
        salary_y = model_total.predict(self.x_test)
        salary_acc = sm.r2_score(self.salary_y_test, salary_y)
        salary_acc = '模型R2评分：' + str(round(salary_acc, 2))
        return [salary_pred_y, salary_acc]

    def PolyRegressor(self):
        #多项式回归
        #1、构造模型
        model_total=pipeline.make_pipeline(PolynomialFeatures(6),LinearRegression())
        #2、模型训练
        model_total.fit(self.x_train,self.salary_y_train)
        #3、模型预测
        total_pred_y=int(model_total.predict(self.pred))
        total_pred_y='Poly总价：'+str(total_pred_y)+'K/月'
        #4、模型评估
        total_y=model_total.predict(self.x_test)
        total_acc=sm.r2_score(self.salary_y_test,total_y)
        total_acc='模型R2评分：'+str(round(total_acc,2))
        return [total_pred_y, total_acc]

    def DecisionTree(self):
        #决策树模型
        # 1、构造模型
        model_total = DecisionTreeRegressor(max_depth=3)
        # 2、模型训练
        model_total.fit(self.x_train, self.salary_y_train)
        # 3、模型预测
        total_pred_y = int(model_total.predict(self.pred))
        total_pred_y = 'Deci总价：' + str(total_pred_y) + 'K/月'
        # 4、模型评估
        salary_y = model_total.predict(self.x_test)
        total_acc = sm.r2_score(self.salary_y_test, salary_y)
        total_acc = '模型R2评分：' + str(round(total_acc, 2))
        return [total_pred_y, total_acc]

    def SVR(self):
        #支持向量机模型
        #1、构造最优模型
        model_total = SVR(kernel='poly',C=2.0,degree=4)
        # 2、模型训练
        model_total.fit(self.x_train, self.salary_y_train)
        # 3、模型预测
        total_pred_y = int(model_total.predict(self.pred))
        total_pred_y = 'SVR总价：' + str(total_pred_y) + '万元'
        # 4、模型评估
        total_y = model_total.predict(self.x_test)
        total_acc = sm.r2_score(self.salary_y_test, total_y)
        total_acc = '模型R2评分：' + str(round(total_acc, 2))
        return [total_pred_y, total_acc]

    def Bagging(self):
        model_total = BaggingRegressor(KNeighborsRegressor(n_neighbors=3),n_estimators=700)
        # 2、模型训练
        model_total.fit(self.x_train, self.salary_y_train)
        # 3、模型预测
        total_pred_y = int(model_total.predict(self.pred))
        total_pred_y = 'Bag总价：' + str(total_pred_y) + '万元'
        # 4、模型评估
        total_y = model_total.predict(self.x_test)
        total_acc = sm.r2_score(self.salary_y_test, total_y)
        total_acc = '模型R2评分：' + str(round(total_acc, 2))

        return [total_pred_y, total_acc]

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
        y_data=self.salary_y_train.values.reshape(-1,1)
        error=model.train(x_data,y_data,epochs=2,show=1,lr=0.01)
        #模型预测
        pred=model.sim(self.pred)[0][0]
        #模型评估
        pred_test_y=model.sim(self.x_test)
        metr=round(sm.r2_score(pred_test_y,self.salary_y_test),2)
        return [pred,metr]

c = MachineLearn('北京', '本科', '3')
# print(c.LineRegressor())
# print(c.PolyRegressor())
print(c.DecisionTree())
# print(c.SVR())
# print(c.Bagging())
# print(c.DNN())