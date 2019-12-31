import os
import random
import numpy as np
import shutil
# 构造卷积神经网络
from keras.layers import Dense, Convolution2D, MaxPool2D, Flatten, Dropout
from keras.models import load_model, Sequential
from keras.preprocessing import image
from .data_gen import DataGenerator

class CatDog(object):
    def __init__(self, file_path):
        self.file_path = file_path  # 预测数据
        # 构造数据
        self.BASE_PATH = os.path.abspath(os.path.dirname(__file__)) + os.sep
        self.train = os.path.join(self.BASE_PATH, 'data', 'catdog', 'train') + os.sep
        # 目标训练地址
        self.target = os.path.join(self.BASE_PATH, 'data', 'cat_dog', 'target') + os.sep
        # 模型地址
        self.model_path = os.path.join(self.BASE_PATH, 'static', 'model') + os.sep
        self.model_path_url = os.path.join(self.BASE_PATH, 'static', 'model', 'my_model_weights.h5') + os.sep

    def ensure_dir(self, dir_path):
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)
            except OSError:
                pass

    def del_target_dir(self):
        # 删除target目录
        target = self.target
        try:
            shutil.rmtree(target)  # 删除整个目录
        except FileNotFoundError:
            print('文件目录：{} 不存在'.format(self.target))

    # 构造数据集
    def init_cat_dog(self, fresh=False):
        # 构造数据集
        if fresh == True:
            try:
                self.del_target_dir()
            except Exception:
                pass

        if not os.path.exists(self.target):
            # 创建保存数据的路径
            self.ensure_dir(self.target + 'train' + os.sep + 'dog' + os.sep)
            self.ensure_dir(self.target + 'train' + os.sep + 'cat' + os.sep)
            self.ensure_dir(self.target + 'test' + os.sep + 'dog' + os.sep)
            self.ensure_dir(self.target + 'tets' + os.sep + 'cat' + os.sep)
            #训练集和测试集分离
            train_list=os.listdir(self.train)
            dogs=[self.train+i for i in train_list if 'dog' in i]
            cats=[self.train+ i for i in train_list if 'cat' in i]
            random.shuffle(dogs)
            random.shuffle(cats)
            cut_size=int(len(dogs)*0.75)
            #todo:构造训练数据
            for dog_file in dogs[:cut_size]:#训练集
                shutil.copyfile(dog_file,self.target+'train'+os.sep+'dog'+os.sep+os.path.basename(dog_file))
            for cat_file in cats[:cut_size]:#训练集
                shutil.copyfile(cat_file,self.target+'train'+os.sep+'cat'+os.sep+os.path.basename(cat_file))
            #todo：构造测试数据
            for dog_file in dogs[cut_size]:#测试集
                shutil.copyfile(dog_file,self.target+'test'+os.sep+'dog'+os.sep+os.path.basename(dog_file))
            for cat_file in cats[:cut_size]:#测试集
                shutil.copyfile(cat_file,self.target+'test'+os.sep+'cat'+os.sep+os.path.basename(cat_file))
        else:
            print('数据已存在，无需重新加载！')

    #读取数据
    def init_data(self,datetype='train'):
        #读取训练数据和测试数据
        datas=[]
        data_path=self.target+datetype+os.sep

        for file in os.listdir(data_path):
            file_path=os.path.join(data_path,file)
            if os.path.isdir(file_path):
                for subfile in os.listdir(file_path):#listdir返回文件名
                    datas.append(os.path.join(file_path,subfile))

        return datas

    def init_mode(self):
        #统一图像尺寸
        img_width=128
        img_height=128
        input_shape=(img_width,img_height,3)
        model=Sequential([
            Convolution2D(32,(3,3),input_shape=input_shape,activation='relu'),
            MaxPool2D(pool_size=(2,2),strides=(2,2),name='pool1'),#池化层
            Convolution2D(64,(3,3),activation='relu'),
            MaxPool2D(pool_size=(2,2),strides=(2,2),name='pool2'),#池化层
            Flatten(),#扁平化
            Dense(64,activation='relu'),
            Dropout(0.5),#随机失活50%
            Dense(2,activation='sigmoid'),
        ])
        #编译模型
        #动量梯度下降算法，交叉熵损失函数，正确率评估模型的指标
        model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
        self.model=model

    #模型训练
    def train_cat_dog(self,fresh=False):
        if fresh==True:#是否需要删除原始模型重新训练
            try:
                shutil.rmtree(self.model_path)
            except Exception:
                pass
        if not os.path.exists(os.path.join(self.model_path,'my_model.h5')):
            self.init_mode()
            #训练数据
            train_datas=self.init_data(datetype='train')
            train_generator=DataGenerator(train_datas,batch_size=32,shuffle=True)
            self.model.fit_generator(train_generator,epochs=15,max_queue_size=10,workers=1,verbose=1)
            self.save_my_model()
        else:
            self.model=self.load_my_model()

    #模型预测
    def pred_cat_dog(self):
        img=image.load_img(self.file_path,target_size=(128,128))
        x=image.img_to_array(img)
        x/=255
        x=np.expand_dims(x,axis=0)
        y=self.model.predict(x)
        pred_index=np.argmax(y)
        if pred_index==0:
            return '识别结果：-<猫咪>-'
        else:
            return '识别结果：-<小狗>-'

    #模型评估
    def eval_my_model(self):
        test_datas=self.init_data(datetype='tets')
        train_generator=DataGenerator(test_datas,batch_size=32,shuffle=True)
        eval_res=self.model.fit_generator(train_generator,max_queue_size=10,workers=1,verbose=1)
        return eval_res

    #保存模型
    def save_my_model(self):
        self.ensure_dir(self.model_path)
        self.model.save(os.path.join(self.model_path,'my_model.h5'))
        self.model.save_weights(os.path.join(self.model_path,'my_model_weights.h5'))

    def load_my_model(self):
        model=load_model(os.path.join(self.model_path,'my_model.h5'))
        model.load_weights(os.path.join(self.model_path,'my_model_weights.h5'))
        return model

# c=CatDog('dog2.jpg')
# c.train_cat_dog()