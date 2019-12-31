import os
import pandas as pd
class pre_data(object):
    def __init__(self):
        #数据保存途径
        self.PATH=os.path.abspath(os.path.dirname(__file__))+os.sep+'data'+os.sep

    def do_data(self):
        #清洗数据，数据离散化
        preprocess_path=self.PATH+'pre_python.csv'#数据处理后的路径
        if not os.path.exists(preprocess_path):
            try:
                file_path=self.PATH+'python.csv'
                df=pd.read_csv(file_path)
                df.loc[:,'pre_worktime']=df.loc[:,'work_time']
            except Exception as e:
                print(e)
                return e
        else:
            return '文件已处理成功{}'.format(preprocess_path)