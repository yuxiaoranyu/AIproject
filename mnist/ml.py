import random
import os
import numpy as np
from PIL import Image
# 创建深度学习模型
from keras.models import Sequential, load_model
from keras.utils import np_utils
from keras.layers.core import Dropout, Dense, Activation
import tensorflow as tf


# 模型保存和加载
class StateModel(object):
    def model_load(self, fn_directory):
        model = load_model(fn_directory)
        return model

    def model_save(self, fn_directory):
        self.model.save(fn_directory)


class MachineLearn(StateModel):
    def __init__(self, file_path):
        # 构造数据
        # todo:构造训练数据
        BATH_PATH = os.path.abspath(os.path.dirname(__file__)) + os.sep
        FILE_PATH = BATH_PATH + 'data' + os.sep + 'mnist.npz'
        f = np.load(FILE_PATH)
        x_train = f['x_train']
        y_train = f['y_train']  # 6000张
        x_test = f['x_test']
        y_test = f['y_test']  # 1000张

        # print(y_train.shape)#(60000,)
        # print(y_test.shape)#(10000,)

        self.img_size = 28 * 28
        self.nb_classes = 10
        self.batch_size = 128  # 批尺寸
        self.nb_epoches = 2  # 迭代次数

        # 特征数据处理
        self.x_train = (x_train.reshape(y_train.shape[0], self.img_size).astype('float32')) / 255
        self.x_test = (x_test.reshape(y_test.shape[0], self.img_size).astype('float32')) / 255
        
        # 标签数据处理
        # 独热编码化
        # print(y_train[:5])
        self.y_train = np_utils.to_categorical(y_train, self.nb_classes)
        self.y_test = np_utils.to_categorical(y_test, self.nb_classes)

        # todo:构造预测数据
        im = Image.open(file_path).resize((28, 28)).convert('L')  # 灰化
        im_arr = np.array(im)
        pred_img = (im_arr.reshape(self.img_size).astype('float32')) / 255
        x_pred = []
        x_pred.append(pred_img)
        self.x_pred = np.array(x_pred)
        # print(self.x_pred)

    # todo:浅层神经网络
    def DNN_keras(self):
        BATH_PATH = os.path.abspath(os.path.dirname(__file__)) + os.sep
        Model_PATH = BATH_PATH + 'static' + os.sep + 'model' + os.sep + 'dnn.h5'
        if os.path.exists(Model_PATH):
            self.model = self.model_load(Model_PATH)
        else:
            # 1、构造模型
            self.model = Sequential([
                Dense(10, input_shape=(self.img_size,), activation='softmax')
            ])
            # 编译模型
            # 动量梯度下降算法，交叉熵损失函数，正确率评估模型的指标
            self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
            # 模型训练
            self.model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.nb_epoches, verbose=1,
                           validation_data=(self.x_test, self.y_test))
            self.model_save(Model_PATH)

        # 4、模型预测
        pred = self.model.predict(self.x_pred)
        pred = np.argmax(pred)
        # 5、模型评估
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)  # 返回误差和得分
        metr = round(score[1], 2)
        metr = 'DNN正确率：' + str(metr)
        pred = '预测结果：' + str(pred)

        return pred, metr,score

    # todo:深度神经网络
    def MLP_keras(self):
        BATH_PATH = os.path.abspath(os.path.dirname(__file__)) + os.sep
        Model_PATH = BATH_PATH + 'static' + os.sep + 'model' + os.sep + 'mlp.h5'
        if os.path.exists(Model_PATH):
            self.model = self.model_load(Model_PATH)
        else:
            # todo：1、构造深度神经网络模型
            self.model = Sequential([
                Dense(512, input_shape=(self.img_size,), activation='sigmoid'),  # 参数数量：718*512+512=401920
                Activation('relu'),
                Dropout(0.5),  # 随机失活
                Dense(256, input_shape=(512,), activation='sigmoid'),  # 参数数量：512*512+512=26256
                Activation('relu'),
                Dropout(0.5),  # 随机失活
                Dense(10, input_shape=(512,), activation='sigmoid'),  # 参数数量：512*10+10=5130
                # Activation('relu'),
                # Dropout(0.5),
                # Dense(2,input_shape=(512,),activation='softmax'),
            ])

            # 2、编译模型
            # 动量梯度下降算法，交叉熵损失函数，正确率评估模型的指标
            self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', weighted_metrics=['accuracy'])
            # 3、模型训练
            self.model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.nb_epoches, verbose=1,
                           validation_data=(self.x_test, self.y_test))
            self.model_save(Model_PATH)

        # 4、模型预测
        pred = self.model.predict(self.x_pred)
        pred = np.argmax(pred)
        # 5、模型评估
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return score
        # metr = round(score[1], 2)
        # metr = 'MLP正确率：' + str(metr)
        # pred = '预测结果：' + str(pred)
        #
        # return pred, metr,score

    def DNN_Tensorflow(self):
        BATH_PATH = os.path.abspath(os.path.dirname(__file__)) + os.sep
        Model_PATH = BATH_PATH + 'static' + os.sep + 'model' + os.sep + 'model.ckpt'

        # 占位符
        x = tf.compat.v1.placeholder('float32', [None, 784])
        y = tf.compat.v1.placeholder('float32', [None, 10])

        # 权重
        W = tf.Variable(tf.zeros([784, 10]))
        # 偏置
        b = tf.Variable(tf.zeros([10]))
        # 预测函数
        y_pred = tf.nn.softmax(tf.matmul(x, W) + b)

        # 损失函数：自定义交叉熵函数
        loss = -tf.reduce_sum(y * tf.math.log(y_pred))
        # 简历梯度下降算法：优化器
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.01)
        # 训练步骤
        train_step = optimizer.minimize(loss)

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())

            if os.path.exists(BATH_PATH + 'static' + os.sep + 'checkpoint'):
                tf.train.Saver().restore(sess, Model_PATH)  # 载入模型
            else:
                # 开始训练
                train_len = self.y_train.shape[0]  # 6000
                for i in range(3000):
                    ram_size1 = random.randint(1, train_len)
                    ram_size2 = ram_size1 + 128
                    if ram_size2 > train_len:
                        ram_size2 = ram_size1
                        ram_size1 = ram_size1 - 128
                    x_train = self.x_train[ram_size1:ram_size2]
                    y_train = self.y_train[ram_size1:ram_size2]
                    sess.run(train_step, feed_dict={x: x_train, y: y_train})

                    if i % 100 == 0:
                        print('第{}次---->'.format(i), sess.run(loss, feed_dict={x: x_train, y: y_train}))
                # 保存模型
                tf.compat.v1.train.Saver().save(sess=sess, save_path=Model_PATH)  # 保存模型
            # 模型评估
            correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))  # equal判断是否预测正确
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 求出正确率的均值
            metr = sess.run(accuracy, feed_dict={x: self.x_test, y: self.y_test})

            pred = sess.run(y_pred, feed_dict={x: self.x_pred})
            pred = np.argmax(pred)
            # 5、模型评估
            metr = 'TensorFlow正确率：' + str(metr)
            pred = '预测结果：' + str(pred)

        return pred, metr


c = MachineLearn("D:\\AI_project\\mnist\\static\\1.png")
# print(c.DNN_keras())
print(c.MLP_keras())
# print(c.DNN_Tensorflow())
