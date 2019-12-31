from .prep import data_preprocessing
import os
from sklearn import ensemble
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import sklearn.svm as svm
from sklearn import cluster
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

class ML(object):
    def __init__(self,sex,initial,age,sibsp,parch,fare,embarked,pclass):
        #todo:构造预测数据
        self.sex=sex
        self.initial=initial
        self.age=age
        self.age_pre()
        self.sibsp=sibsp
        self.parch=parch
        self.fare=fare
        self.fare_pre()
        self.embarked=embarked
        self.pclass=pclass

        self.family_size=self.sibsp+self.parch
        if self.family_size>0:
            self.alone=0
        else:
            self.alone=1
        self.pred=[[self.pclass,self.sibsp,self.parch,self.initial,self.age,self.family_size,self.alone,self.embarked,self.sex,self.fare]]
        #todo：构造训练数据
        PATH=os.path.abspath(os.path.dirname(__file__))+os.path.sep
        file_path=PATH+'data'+os.path.sep+'titanic.csv'
        save_path=PATH+'data'+os.path.sep+'mydata.csv'
        if not os.path.exists(file_path,save_path):
            #处理数据后
            data_preprocessing(file_path,save_path)
        #构造数据
        data=pd.read_csv(save_path)
        data=data[data.columns[1:]]
        #训练集和测试集分离
        x=data[data.columns[1:]]
        y=data[data.columns[0]]
        x,y=shuffle(x,y,random_state=7)#随机打乱
        #提升泛化能力，shuffle
        self.X,self.Y=shuffle(x,y,random_state=8)
        #训练集和测试集切分
        self.X_train,self.X_test,self.Y_train,self.Y_test=train_test_split(self.X,self.Y,test_size=0.2,random_state=8)


    def age_pre(self):
        # todo:年龄离散化
        # min 0.420000
        # 25 % 22.000000
        # 50 % 30.000000
        # 75 % 36.000000
        # max 80.000000
        _min = 0.42
        _tf = 22
        _mid = 30
        _sf = 36
        _max = 80
        # 预测数据年龄离散化
        if self.age <= _tf:
            self.age = 0
        elif self.age <= _mid:
            self.age = 1
        elif self.age <= _sf:
            self.age = 3
        else:
            self.age = 3
        return self.age

    def fare_pre(self):
        # todo:费用离散化
        # print(data["Fare"].describe())
        # min        0.000000
        # 25%        7.910400
        # 50%       14.454200
        # 75%       31.000000
        # max      512.329200
        _min = 0.000000
        _tf = 7.910400
        _mid = 14.454200
        _sf = 31.000000
        _max = 512.329200
        # 预测数据票价离散化
        if self.fare <= _tf:
            self.fare = 0
        elif self.fare <= _mid:
            self.fare = 1
        elif self.fare <= _sf:
            self.fare = 2
        else:
            self.fare = 3
        return self.fare

    def KNN(self):
        # 构造数据
        model = KNeighborsClassifier(n_neighbors=3)
        # 模型训练
        model.fit(self.X_train, self.Y_train)
        # 模型预测及评估
        pred_test_y = model.predict(self.X_test)
        sc = round(metrics.accuracy_score(self.Y_test, pred_test_y), 2)
        # 接收前端传过来的数据进行模型应用
        pred = model.predict(self.pred)[0]

        if pred == 0:
            pred = '-死亡-'
        else:
            pred = '-存活-'
        metr = '模型正确率' + str(sc)
        pred = 'KNN结果：' + str(pred)
        return pred, metr

    def LogsticRegression(self):
        # 构造数据
        # 构造逻辑回归模型
        model = LogisticRegression()
        # 模型训练
        model.fit(self.X_train, self.Y_train)
        # 模型预测及评估
        pred_test_y = model.predict(self.X_test)
        sc = round(metrics.accuracy_score(self.Y_test, pred_test_y), 2)
        # 接收前端传过来的数据进行模型应用
        pred = model.predict(self.pred)[0]

        if pred == 0:
            pred = '-死亡-'
        else:
            pred = '-存活-'
        metr = '模型正确率' + str(sc)
        pred = 'Logi结果：' + str(pred)
        return pred, metr

    def DecisionTree(self):
        # 超参数调整
        hyper = {'criterion': ['gini', 'entropy'], 'max_depth': range(1, 6)}  # 参数选择
        scoring_func = metrics.make_scorer(metrics.accuracy_score)  # 评估指标
        kfold = KFold(n_splits=5)  # 交叉验证
        model = DecisionTreeClassifier()  # 决策树模型
        grid = GridSearchCV(estimator=model, param_grid=hyper, scoring=scoring_func, cv=kfold, verbose=True)
        grid.fit(self.X, self.Y)
        acc = grid.best_score_
        reg = grid.best_estimator_
        lis = []
        for key in hyper.keys():
            n = reg.get_params()[key]  # 取出最佳参数
            lis.append(n)

        # 构造模型
        model = DecisionTreeClassifier(criterion=lis[0], max_depth=lis[1])
        # 模型训练
        model.fit(self.X_train, self.Y_train)
        # 模型评估
        pred_test_y = model.predict(self.X_test)
        metr = round(metrics.accuracy_score(pred_test_y, self.Y_test), 2)
        # 模型预测
        pred = model.predict(self.pred)[0]

        if pred == 0:
            pred = '-死亡-'
        else:
            pred = '-存活-'
        metr = '模型正确率' + str(metr)
        pred = 'Des结果：' + str(pred)
        return pred, metr

    def RandomForest(self):
        # 构造模型
        model = RandomForestClassifier(criterion='gini', max_depth=3, n_estimators=30)
        # 模型训练
        model.fit(self.X_train, self.Y_train)
        # 模型评估
        pred_test_y = model.predict(self.X_test)
        metr = round(metrics.accuracy_score(pred_test_y, self.Y_test), 2)
        # 模型预测
        pred = model.predict(self.pred)[0]

        if pred == 0:
            pred = '-死亡-'
        else:
            pred = '-存活-'
        metr = '模型正确率' + str(metr)
        pred = 'RF结果：' + str(pred)
        return pred, metr

    def SVM(self):
        # 构造模型
        model = svm.SVC(gamma=0.01, C=0.1, kernel='linear')
        # 模型训练
        model.fit(self.X_train, self.Y_train)
        # 模型评估
        pred_test_y = model.predict(self.X_test)
        metr = round(metrics.accuracy_score(pred_test_y, self.Y_test), 2)
        # 模型预测
        pred = model.predict(self.pred)[0]

        if pred == 0:
            pred = '-死亡-'
        else:
            pred = '-存活-'
        metr = '模型正确率' + str(metr)
        pred = 'SVM结果：' + str(pred)
        return pred, metr

    def KMeans(self):
        # 构建聚类模型：凝聚层次算法
        model = cluster.KMeans(n_clusters=2)
        # 模型训练
        model.fit(self.X)
        # 模型评估
        metr = round(metrics.silhouette_score(self.X, model.labels_), 2)
        # 模型预测
        pred = model.predict(self.pred)[0]

        if pred == 0:
            pred = '-死亡-'
        else:
            pred = '-存活-'
        metr = '模型轮廓系数' + str(metr)
        pred = 'KMeans结果：' + str(pred)
        return pred, metr

    def Bagging(self):
        # 构造模型
        model = ensemble.BaggingClassifier(KNeighborsClassifier(n_neighbors=3), n_estimators=700)
        # 模型训练
        model.fit(self.X_train, self.Y_train)
        # 模型评估
        pred_test_y = model.predict(self.X_test)
        metr = round(metrics.accuracy_score(pred_test_y, self.Y_test), 2)
        # 模型预测
        pred = model.predict(self.pred)[0]

        if pred == 0:
            pred = '-死亡-'
        else:
            pred = '-存活-'
        metr = '模型正确率' + str(metr)
        pred = 'Bagging结果：' + str(pred)
        return pred, metr

    def Adaboost(self):
        # 构造模型
        model = ensemble.AdaBoostClassifier(n_estimators=700, random_state=7)
        # 模型训练
        model.fit(self.X_train, self.Y_train)
        # 模型评估
        pred_test_y = model.predict(self.X_test)
        metr = round(metrics.accuracy_score(pred_test_y, self.Y_test), 2)
        # 模型预测
        pred = model.predict(self.pred)[0]

        if pred == 0:
            pred = '-死亡-'
        else:
            pred = '-存活-'
        metr = '模型正确率' + str(metr)
        pred = 'Adaboost结果：' + str(pred)
        return pred, metr