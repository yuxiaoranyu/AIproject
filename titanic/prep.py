import numpy as np
import pandas as pd


def data_preprocessing(read_directory,save_directory):

    data = pd.read_csv(read_directory)
    # print(data["Age"].describe())

    #todo:空值处理：年龄空值，登录港口空值
    #众数填充
    data["Embarked"].fillna("S",inplace=True)

    data["Initial"] = 0 #新增一列索引为Initial的特征列
    for i in data:
        data["Initial"] = data["Name"].str.extract("(\w+)\.")
    # print(set(data.Initial))
    #替换称呼
    data['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],
                            ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
    #求出平均数
    initial_age_mean = data.groupby("Initial")["Age"].mean()
    #年龄缺失值的填充
    data.loc[(data.Age.isnull())&(data.Initial=="Master"),"Age"] = 5
    data.loc[(data.Age.isnull())&(data.Initial=="Miss"),"Age"] = 22
    data.loc[(data.Age.isnull())&(data.Initial=="Mr"),"Age"] = 33
    data.loc[(data.Age.isnull())&(data.Initial=="Mrs"),"Age"] = 36
    data.loc[(data.Age.isnull())&(data.Initial=="Other"),"Age"] = 46

    #todo:年龄的离散化
    data["Age_new"] = 0
    # print(data["Age"].describe())
    data.loc[data["Age"] <=22,"Age_new"] = 0
    data.loc[(data["Age"] > 22)&(data["Age"] <=30),"Age_new"] = 1
    data.loc[(data["Age"] > 30)&(data["Age"] <=36),"Age_new"] = 2
    data.loc[(data["Age"] > 36)&(data["Age"] <=58),"Age_new"] = 3
    data.loc[data["Age"] > 58,"Age_new"] = 4

    #todo:家庭人数
    data["Family_size"] = data["Parch"] + data["SibSp"]
    data["Alone"] = 0 #独身，默认不独身
    data.loc[data["Family_size"]==0,"Alone"] = 1

    #todo:登陆地
    data["Embarked_num"] = 0
    data.loc[data["Embarked"]=="S","Embarked_num"] = 2
    data.loc[data["Embarked"]=="C","Embarked_num"] = 1
    data.loc[data["Embarked"]=="Q","Embarked_num"] = 0

    #todo:性别
    #标签编码
    data["Sex_num"] = 0 #默认为0，男性
    data.loc[data["Sex"]=="female","Sex_num"] = 1

    #独热编码
    # data["Sex_male"] = 0
    # data["Sex_female"] = 0
    # data.loc[data["Sex"]=="female","Sex_female"] = 1
    # data.loc[data["Sex"]=="male","Sex_male"] = 1


    #todo:票价
    # print(data["Fare"].describe())
    # min        0.000000
    # 25%        7.910400
    # 50%       14.454200
    # 75%       31.000000
    # max      512.329200
    data["Fare_new"] = 0
    data.loc[data["Fare"] <=8,"Fare_new"] = 0
    data.loc[(data["Fare"] > 8)&(data["Age"] <=15),"Fare_new"] = 1
    data.loc[(data["Fare"] > 15)&(data["Age"] <=31),"Fare_new"] = 2
    data.loc[(data["Fare"] > 31)&(data["Age"] <=271.5),"Fare_new"] = 3
    data.loc[data["Fare"] > 271.5,"Fare_new"] = 4

    #todo：称呼
    # print(set(data["Initial"]))
    data["Initial"].replace(['Master', 'Other','Miss', 'Mrs', 'Mr'],
                            [0,1,2,3,4],inplace=True)

    data.drop(['PassengerId','Name','Sex','Age','Ticket','Fare','Cabin','Embarked'],axis=1,inplace=True)

    #保存数据
    data.to_csv(save_directory)

    return save_directory  #返回保存成功后的路径