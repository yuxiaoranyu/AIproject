from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble
from sklearn.svm import SVC
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder,scale
import neurolab
import warnings
warnings.filterwarnings('ignore')


class MachineLearn(object):
    def __init__(self,file_path,pca_k):
        #1、构造数据
        #todo：训练集
        BASE_PATH=os.path.abspath(os.path.dirname(__file__))+os.sep
        PATH=BASE_PATH+'att_faces'+os.sep
        x=[]
        y=[]
        #将照片导入numpy数组，将像素矩阵替换为向量
        for dir_path,dir_names,file_names in os.walk(PATH):
            for fn in file_names:
                if fn[-3:]=='pgm':
                    image_filename=os.path.join(dir_path,fn)
                    #图像处理
                    img=Image.open(image_filename).convert('L')#灰化
                    im_arr=np.array(img)
                    # print(im_arr.shape)#112*92=10304
                    x_d=scale(im_arr.reshape(10304).astype('float32'))
                    x.append(x_d)
                    y.append(dir_path)
        # print(y)

        x=np.array(x)
        #训练集和测试集分离
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(x,y,test_size=0.25,random_state=7)
        #todo：pca降维
        pca=PCA(n_components=pca_k)
        self.x_train=pca.fit_transform(self.x_train)
        self.x_test=pca.transform(self.x_test)
        # print(self.x_train.shape)#(297, 150)
        # print(self.x_test.shape)#(100, 150)
        #处理标签集
        yy_train=[]
        for i in range(len(self.y_train)):
            yy=int(str(self.y_train[i].split('\\')[-1])[1:])
            yy_train.append(yy)

        self.y_train=np.array(yy_train).reshape(-1,1)
        yy_test=[]
        for i in range(len(self.y_test)):
            yy=int(str(self.y_test[i].split('\\')[-1])[1:])
            yy_test.append(yy)
        self.y_test=np.array(yy_test).reshape(-1,1)
        # print(self.y_train[:5])
        # print(self.y_test[:5])

        #神经网络标签：独热编码
        self.one=OneHotEncoder(sparse=False,dtype=int)
        self.nn_y_train=self.one.fit_transform(self.y_train)
        self.nn_y_test=self.one.transform(self.y_test)

        #todo：构造测试集数据
        im=Image.open(file_path).convert('L')
        im_arr=np.array(im)
        pre_face=scale(im_arr.reshape(-1).astype('float32'))
        x_pred=[pre_face]
        x_pred=np.array(x_pred)
        self.pred=pca.transform(x_pred)#降维处理

    def KNN(self):
        #1、模型构造
        model=KNeighborsClassifier(n_neighbors=1)
        #2、模型训练
        model.fit(self.x_train,self.y_train)
        #3、模型预测
        pred=model.predict(self.pred)[0]
        pred='KNN预测：'+'s'+str(pred)
        #4、模型评估
        pred_test_y=model.predict(self.x_test)
        metr=round(metrics.accuracy_score(self.y_test,pred_test_y),2)
        metr='正确率：'+str(metr)
        return pred,metr

    def LogsticRegressor(self):
        #1、模型构造
        #1-bfgs：共轭梯度法，还有三种算法，liblinear,newton-cg,sag
        #分类选择："ovr","multinomial",todo:multinomial一般用在多元逻辑回归上
        model=LogisticRegression(multi_class='multinomial',solver='lbfgs')
        #2、模型训练
        model.fit(self.x_train,self.y_train)
        #3、模型预测
        pred=model.predict(self.pred)[0]
        pred='Log预测：'+'s'+str(pred)
        #4、模型评估
        pred_test_y=model.predict(self.x_test)
        metr=round(metrics.accuracy_score(self.y_test,pred_test_y),2)
        metr='正确率：'+str(metr)
        return pred,metr

    def DecisionTree(self):
        #1、模型构造
        model=DecisionTreeClassifier(criterion='entropy',max_depth=40)
        #2、模型训练
        model.fit(self.x_train,self.y_train)
        #3、模型预测
        pred=model.predict(self.pred)[0]
        pred='Dec预测：'+'s'+str(pred)
        #4、模型评估
        pred_test_y=model.predict(self.x_test)
        metr=round(metrics.accuracy_score(self.y_test,pred_test_y),2)
        metr='正确率：'+str(metr)
        return pred,metr

    def RandomFroest(self):
        #1、模型构造
        model=ensemble.RandomForestClassifier(max_depth=30,criterion='entropy',n_estimators=20,random_state=7)
        #2、模型训练
        model.fit(self.x_train,self.y_train)
        #3、模型预测
        pred=model.predict(self.pred)[0]
        pred='RF预测：'+'s'+str(pred)
        #4、模型评估
        pred_test_y=model.predict(self.x_test)
        metr=round(metrics.accuracy_score(self.y_test,pred_test_y),2)
        metr='正确率：'+str(metr)
        return pred,metr

    def SVM(self):
        #1、模型构造
        model=SVC(kernel='linear',C=0.1,gamma=0.01)
        #2、模型训练
        model.fit(self.x_train,self.y_train)
        #3、模型预测
        pred=model.predict(self.pred)[0]
        pred='SVM预测：'+'s'+str(pred)
        #4、模型评估
        pred_test_y=model.predict(self.x_test)
        metr=round(metrics.accuracy_score(self.y_test,pred_test_y),2)
        metr='正确率：'+str(metr)
        return pred,metr

    def Bagging(self):
        #1、模型构造
        model=ensemble.BaggingClassifier(KNeighborsClassifier(n_neighbors=1),n_estimators=20)
        #2、模型训练
        model.fit(self.x_train,self.y_train)
        #3、模型预测
        pred=model.predict(self.pred)[0]
        pred='Bag预测：'+'s'+str(pred)
        #4、模型评估
        pred_test_y=model.predict(self.x_test)
        metr=round(metrics.accuracy_score(self.y_test,pred_test_y),2)
        metr='正确率：'+str(metr)
        return pred,metr

    def DNN_python(self):
        #神经网络模型预测
        nn_minmax=[]
        for i in range(self.x_train.shape[-1]):
            min_x=self.x_train.T[i].min()
            max_x=self.x_train.T[i].max()
            nn_minmax.append([min_x,max_x])
        nn_minmax=np.array(nn_minmax)#值域

        #todo:构造单层神经网络模型
        model=neurolab.net.newp(nn_minmax,40)
        #todo:构造深层神经网络模型
        # model=neurolab.net.newff(nn_minmax,[512,40])
        # model.trainf=neurolab.train.train_gd#设置为梯度下降

        #模型训练
        model.train(self.x_train,self.nn_y_train,epochs=1000,show=5,goal=0.01)
        #模型预测
        pred=model.sim(self.pred)
        pred=pred.argmax()+1
        pred='DNN预测：'+'s'+str(pred)
        #模型评估
        pred_y=model.sim(self.x_test)
        pred_yy=[]
        for i in pred_y:
            deco=i.argmax()+1
            pred_yy.append(deco)
        pred_yy=np.array(pred_yy).reshape(-1,1)
        metr=round(metrics.accuracy_score(pred_yy,self.y_test),2)
        metr='正确率：'+str(metr)

        return pred,metr


# c=MachineLearn('./7.pgm',150)
# print(c.KNN())
# print(c.LogsticRegressor())
# print(c.DecisionTree())
# print(c.RandomFroest())
# print(c.SVM())
# print(c.Bagging())
# print(c.DNN_python())





















