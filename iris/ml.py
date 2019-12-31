import os
import pandas as pd
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import sklearn.svm as svm
from sklearn import cluster
import warnings
warnings.filterwarnings('ignore')


class MachineLearn(object):
    """
    sepal length:花萼长度，单位为cm;
    sepal width:花萼宽度，单位为cm;
    petal length:花瓣长度，单位为cm;
    petal width:花瓣宽度，单位为cm;
    Iris-setosa：山鸢尾
    Iris-versicolor：杂色鸢尾
    Iris-virginica：维吉尼亚鸢尾
    """
    def __init__(self):
        #构造数据
        PATH = os.path.abspath(os.path.dirname(__file__)) + os.sep #当前文件的绝对路径
        # print(PATH)
        data = pd.read_csv(PATH+"data"+os.sep+"iris.data",names=["sepal_length","sepal_width","petal_length","petal_width","class"])
        X = data.loc[:,["petal_width"]]
        Y = data.loc[:,["petal_length"]]
        #构造线性回归的数据
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(X,Y,test_size=0.2,random_state=8)

        #分类预测数据
        cls_X = data.loc[:, ["sepal_length","sepal_width","petal_length","petal_width"]]
        cls_Y = data.loc[:, ["class"]]
        cls_X,cls_Y = shuffle(cls_X,cls_Y,random_state=8)
        self.cls_x_train,self.cls_x_test,self.cls_y_train,self.cls_y_test = train_test_split(cls_X,cls_Y,test_size=0.2,random_state=8)
        # print(self.cls_x_train.shape)
        # print(self.cls_x_test.shape)
        # print(self.cls_y_train.shape)
        # print(self.cls_y_test.shape)

        #无监督学习数据
        self.kmean_data = data.loc[:, ["sepal_length","sepal_width","petal_length","petal_width"]]

    #线性回归
    def LineRegression(self,petal_width):
        """构造线性回归模型预测"""
        #构造模型
        model = LinearRegression()
        # 模型训练
        model.fit(self.x_train,self.y_train)
        # 模型预测及评估
        pred_test_y = model.predict(self.x_test)
        sc = round(metrics.r2_score(self.y_test,pred_test_y),2)
        # 接收前端传过来的数据进行模型应用
        pred = round(model.predict(petal_width)[0][0],2)

        return f"花瓣长度：{pred}---->r2得分：{sc}"

    #多项式回归
    def PolyRegression(self,petal_width):
        model = pipeline.make_pipeline(PolynomialFeatures(5),LinearRegression()) #构造最高次幂为5词的多项式回归
        model.fit(self.x_train,self.y_train)
        # 模型预测及评估
        pred_test_y = model.predict(self.x_test)
        sc = round(metrics.r2_score(self.y_test, pred_test_y), 2)
        # 接收前端传过来的数据进行模型应用
        pred = round(model.predict(petal_width)[0][0], 2)

        return f"花瓣长度：{pred}---->r2得分：{sc}"

    #KNN
    def KNN(self,k_max=1,pred=[[1,1,1,1]]):
        #构造数据
        #todo:超参数调整
        # hyper = {"n_neighbors":list(range(1,k_max,1))}
        # model = GridSearchCV(estimator=KNeighborsClassifier(),param_grid=hyper,verbose=True)
        model = KNeighborsClassifier(n_neighbors=8)
        #模型训练
        model.fit(self.cls_x_train,self.cls_y_train)
        # print(model.best_estimator_) #看一下最优参数设置
        # 模型预测及评估
        pred_test_y = model.predict(self.cls_x_test)
        sc = round(metrics.accuracy_score(self.cls_y_test, pred_test_y), 2)
        # 接收前端传过来的数据进行模型应用
        pred = model.predict(pred)[0]
        metr = "模型正确率" + str(sc)
        pred = "KNN结果：" + str(pred)
        return pred,metr

    #逻辑回归
    def LogsticRegression(self,pred=[[1,1,1,1]]):
        #1、构造数据
        #2、构造逻辑回归模型
        model = LogisticRegression()
        #3、模型训练
        model.fit(self.cls_x_train,self.cls_y_train)
        # 模型预测及评估
        pred_test_y = model.predict(self.cls_x_test)
        sc = round(metrics.accuracy_score(self.cls_y_test, pred_test_y), 2)
        # 接收前端传过来的数据进行模型应用
        pred = model.predict(pred)[0]
        metr = "模型正确率" + str(sc)
        pred = "Logi结果：" + str(pred)
        return pred, metr

    #决策树
    def DecisonTree(self,pred=[[1,1,1,1]]):
        #构造数据
        #todo:超参数调整
        # max_de = np.asarray(range(2,5,1))
        # cri = ["gini","entropy"]
        # hyper = {"max_depth":max_de,"criterion":cri}
        # model = GridSearchCV(estimator=DecisionTreeClassifier(),param_grid=hyper,verbose=True)
        model = DecisionTreeClassifier(criterion='gini', max_depth=3)
        #模型训练
        model.fit(self.cls_x_train,self.cls_y_train)
        # print(model.best_estimator_) #看一下最优参数设置
        # 模型预测及评估
        pred_test_y = model.predict(self.cls_x_test)
        sc = round(metrics.accuracy_score(self.cls_y_test, pred_test_y), 2)
        # 接收前端传过来的数据进行模型应用
        pred = model.predict(pred)[0]
        metr = "模型正确率" + str(sc)
        pred = "Deci结果：" + str(pred)
        return pred,metr

    #随机森林
    def RandomForest(self,pred=[[1,1,1,1]]):
        #构造数据
        #todo:超参数调整
        # cri = ["gini", "entropy"]
        # et = np.asarray(range(1, 20, 1))
        # hyper = {"criterion": cri, "n_estimators": et}
        # model = GridSearchCV(estimator=RandomForestClassifier(max_depth=3,random_state=8), param_grid=hyper, verbose=True)

        model = RandomForestClassifier(criterion='gini', n_estimators=5,max_depth=3,random_state=8)
        #模型训练
        model.fit(self.cls_x_train,self.cls_y_train)
        # print(model.best_estimator_) #看一下最优参数设置
        # 模型预测及评估
        pred_test_y = model.predict(self.cls_x_test)
        sc = round(metrics.accuracy_score(self.cls_y_test, pred_test_y), 2)
        # 接收前端传过来的数据进行模型应用
        pred = model.predict(pred)[0]
        metr = "模型正确率" + str(sc)
        pred = "RF结果：" + str(pred)
        return pred,metr

    #支持向量机
    def SVM(self,pred=[1,1,1,1]):
        #构造最优模型
        # gamma = list(np.arange(0.1,1,0.1))
        # C = list(np.arange(1,10,1))
        # kernel = ['linear', 'poly', 'rbf', 'sigmoid']
        # hyper = {"gamma":gamma,"C":C,"kernel":kernel}
        # model = GridSearchCV(estimator=svm.SVC(),param_grid=hyper,verbose=True)
        model = svm.SVC(C=2,gamma=0.5, kernel='rbf',)
        model.fit(self.cls_x_train,self.cls_y_train)
        # print(model.best_estimator_)
        # 模型预测及评估
        pred_test_y = model.predict(self.cls_x_test)
        sc = round(metrics.accuracy_score(self.cls_y_test, pred_test_y), 2)
        # 接收前端传过来的数据进行模型应用
        pred = model.predict(pred)[0]
        metr = "模型正确率" + str(sc)
        pred = "SVM结果：" + str(pred)
        return pred, metr

    #k-means聚类
    def Kmeans(self,pred=[[1,1,1,1]]):
        #无监督学习，构造特征及合不需要标签
        #构造无监督学习模型
        model = cluster.MeanShift()
        #模型训练
        model.fit(self.kmean_data)
        #模型预测
        pred = model.predict(pred)

        if pred == [1]:
            pred = "Iris-setosa"
        elif pred ==[0]:
            pred = "Iris-versicolor"
        else:
            pred = "Iris-versicolor"


        #轮廓系数评估聚类算法
        metr = round(metrics.silhouette_score(self.kmean_data,model.labels_),2)
        metr = "轮廓系数：" + str(metr)
        pred = "KM结果：" + str(pred)
        return pred,metr


# c = MachineLearn()
# #
# print(c.LineRegression(2))
# print(c.KNN(30,[[5,3,1,0.5],]))
# print(c.LogsticRegression([[5,3,1,0.5],]))
# print(c.DecisonTree([[5,3,1,0.5],]))
# print(c.RandomForest([[5,3,1,0.5],]))
# print(c.SVM([[6.0,2.2,4.0,1.0],]))
# print(c.Kmeans([[6.0,2.2,4.0,1.0],]))







