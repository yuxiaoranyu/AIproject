import os
from sklearn import ensemble
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
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


class MachineLearn(object):
    def __init__(self, sex, age, fare, alone):
        self.sex = sex
        self.age = age
        self.fare = fare
        self.alone = alone
        # 构造训练数据
        PATH = os.path.abspath(os.path.dirname(__file__)) + os.path.sep
        file_path = PATH + 'data' + os.path.sep + 'titanic.csv'
        data_train = pd.read_csv(file_path)
        # 数据处理
        # 填充空值
        data_train["Initial"] = 0  # 增加一列，获取名字前的称呼
        data_train["Initial"] = data_train["Name"].str.extract("(\w+)\.")  # 分组提取，通配符.要转义
        # print(set(data_train["Initial"]))
        # {'Master', 'Major', 'Capt', 'Miss', 'Don', 'Mlle', 'Rev', 'Jonkheer', 'Lady', 'Mrs', 'Mme', 'Ms', 'Countess','Col', 'Dr', 'Sir', 'Mr'}
        data_train["Initial"].replace(["Mlle", 'Mme', 'Ms'], ["Miss", "Miss", "Miss"], inplace=True)
        data_train["Initial"].replace(["Lady", 'Countess'], ["Mrs", "Mrs"], inplace=True)
        data_train["Initial"].replace(["Jonkheer", 'Col', 'Rev'], ["other", "other", "other"], inplace=True)
        data_train["Initial"].replace(["Major", 'Capt', 'Sir', "Don", "Dr"], ["Mr", "Mr", "Mr", "Mr", "Mr"],
                                      inplace=True)
        # 根据称呼分组的平均年龄
        # print(data_train.groupby("Initial")["Age"].mean())
        # Master 4.574167 Miss 21.860000 Mr 32.739609 Mrs 35.981818 other 45.888889
        # 填充年龄空值
        data_train["Age_p"] = data_train["Age"]
        data_train.loc[(data_train["Age"].isnull()) & (data_train["Initial"] == "Master"), "Age_p"] = 5
        data_train.loc[(data_train["Age"].isnull()) & (data_train["Initial"] == "Miss"), "Age_p"] = 22
        data_train.loc[(data_train["Age"].isnull()) & (data_train["Initial"] == "Mr"), "Age_p"] = 33
        data_train.loc[(data_train["Age"].isnull()) & (data_train["Initial"] == "Mrs"), "Age_p"] = 36
        data_train.loc[(data_train["Age"].isnull()) & (data_train["Initial"] == "other"), "Age_p"] = 46

        # todo：是否小孩
        data_train['child_p'] = (data_train['Age_p'] <= 10).astype(int)  # 小于等于10岁的为小孩
        # 构建预测数据新的特征列
        if self.age <= 10:
            self.child = 1
        else:
            self.child = 0

        # todo:年龄离散化
        # print(data_train["Age_p"].describe())
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
        data_train.loc[(data_train['Age_p'] > _min) & (data_train['Age_p'] <= _tf), 'Age_p'] = 0
        data_train.loc[(data_train['Age_p'] > _tf) & (data_train['Age_p'] <= _mid), 'Age_p'] = 1
        data_train.loc[(data_train['Age_p'] > _mid) & (data_train['Age_p'] <= _sf), 'Age_p'] = 2
        data_train.loc[(data_train['Age_p'] > _sf) & (data_train['Age_p'] <= _max), 'Age_p'] = 3

        # 预测数据年龄离散化
        if self.age <= _tf:
            self.age = 0
        elif self.age <= _mid:
            self.age = 1
        elif self.age <= _sf:
            self.age = 2
        else:
            self.age = 3

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
        data_train['Fare_p'] = 0
        data_train.loc[(data_train['Fare'] > _min) & (data_train['Fare'] <= _tf), 'Fare_p'] = 0
        data_train.loc[(data_train['Fare'] > _tf) & (data_train['Fare'] <= _mid), 'Fare_p'] = 1
        data_train.loc[(data_train['Fare'] > _mid) & (data_train['Fare'] <= _sf), 'Fare_p'] = 2
        data_train.loc[(data_train['Fare'] > _sf) & (data_train['Fare'] <= _max), 'Fare_p'] = 3

        # 预测数据票价离散化
        if self.fare <= _tf:
            self.fare = 0
        elif self.fare <= _mid:
            self.fare = 1
        elif self.fare <= _sf:
            self.fare = 2
        else:
            self.fare = 3

        # todo:性别编码化
        data_train['Sex_p'] = 0  # 新生成一列用来存储性别
        data_train.loc[data_train['Sex'] == 'famale', 'Sex_p'] = 1
        # 预测数据离散化
        if self.sex == 'male':
            self.sex = 0
        else:
            self.sex = 1

        # todo:是否独身
        # 方法1：
        # data_train['family_size']=0
        # data_train['family_size']=data_train['Sibsp']+data_train['Parch']
        # data_train['alone_p']=0
        # data_train.loc[data_train['family_size']>0,'alone_p']=1
        # 方法2
        data_train['alone_p'] = ((data_train['SibSp'] + data_train['Parch']) > 0).astype(int)

        # 要训练的数据
        train_df = data_train.filter(regex='Survived|.*_p')

        X = train_df.iloc[:, 1:]  # 特征集
        Y = train_df.iloc[:, 0]  # 标签集
        # 提升繁华能力，shuffle
        self.X, self.Y = shuffle(X, Y, random_state=8)
        # 训练集和测试集的切分
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.2,
                                                                                random_state=8)
        # 预测数据pred
        self.pred = [[self.age, self.child, self.fare, self.sex, self.alone]]

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
        metr = '模型正确率：' + str(sc)
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
        metr = '模型正确率：' + str(sc)
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
        metr = '模型正确率：' + str(metr)
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
        metr = '模型正确率：' + str(metr)
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
        metr = '模型正确率：' + str(metr)
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
        metr = '模型轮廓系数：' + str(metr)
        pred = 'Kmeans结果：' + str(pred)
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
        metr = '模型正确率：' + str(metr)
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


c = MachineLearn('female', 24, 500, 0)
# print(c.KNN())
# print(c.LogsticRegression())
# print(c.DecisionTree())
# print(c.RandomForest())
# print(type(c.RandomForest()))
# print(c.SVM())
# print(c.Kmeans())
# print(type(c.Kmeans()))
# print(c.Bagging())
# print(c.Adaboost())
