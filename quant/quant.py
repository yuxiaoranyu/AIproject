import tushare as ts
import pandas as pd
import numpy as np
import pymysql
from sqlalchemy import create_engine
import os
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression



class Quant(object):
    def __init__(self):
        self.PATH = os.path.abspath(os.path.dirname(__file__)) + os.sep + 'data' + os.sep

    def get_realtime_stock_info(self, code):
        codes = code.split(' ')
        df = ts.get_realtime_quotes(codes)
        df = df[['code', 'name', 'open', 'price', 'high', 'low', 'volume', 'amount', 'pre_close']]
        results = []
        for i in range(len(df)):
            clo_name = df.ix[i, ['name']].to_dict()
            clo_code = df.ix[i, ['code']].to_dict()
            clo_open = df.ix[i, ['open']].to_dict()
            clo_price = df.ix[i, ['price']].to_dict()
            clo_pre_close = df.ix[i, ['pre_close']].to_dict()
            stock_real_info_dict = dict(clo_name, **clo_code, **clo_open, **clo_price, **clo_pre_close)
            results.append(stock_real_info_dict)
        return results

    def catch_data(self, code, start_time, end_time):
        df = ts.get_k_data(code, start=start_time, end=end_time)
        engine = create_engine('mysql+pymysql://root:root@localhost/quant?charset=utf8')
        df.to_sql('stock' + code, engine, if_exists='replace', index=False)
        file_name = 'stock' + code
        return file_name

    def out_csv(self, code):
        conn = pymysql.connect('localhost', port=3306, user='root', passwd='root', db='quant')
        sql = f'select date,open,close from stock{code}'
        result = pd.read_sql(sql, conn)
        stock_path = self.PATH + 'stock' + code + '.csv'
        result.to_csv(stock_path, encoding='UTF-8', header=False)
        return stock_path

    def dodata(self, code):
        stock_path = self.PATH + 'stock' + code + '.csv'
        self.spy = pd.read_csv(stock_path, names=['date', 'open', 'close'])
        # 预测当天收盘价，从当前日的前20天建模，生成20列特征（前20天的收盘价）
        for i in range(1, 21):
            self.spy.loc[:, f'close Minus{i}'] = self.spy['close'].shift(i)  # 行索引的是数字未发生改变，值向下移动i位

        self.sp20 = self.spy.filter(regex='date|^close$|close Minus.*')
        self.sp20 = self.sp20.iloc[20:, :]  # 前20行数据存在空值，从2019-01-01向后推20天做统计
        # 数据分割
        x = self.sp20.iloc[:, 2:]  # 训练集
        y = self.sp20.iloc[:, 1].shift(-1)  # 标签集close

        # 80%做训练，20%做测试
        train_size = int(len(y) * 0.8)
        self.x_train = x[:train_size]
        self.x_test = x[train_size:]
        self.y_train = y[:train_size]
        self.y_test = y[train_size:]
        return len(self.x_train), len(self.x_test)

    def SVR(self, code):
        self.dodata(code)
        # 构造模型
        model = SVR(kernel='linear')
        # 训练模型
        model.fit(self.x_train, self.y_train)
        # 模型预测
        self.preds = model.predict(self.x_test)
        # 今天的收盘价预测
        today_pred = round(self.preds[self.preds.shape[0] - 1], 2)
        today_pred = 'SVM预测价格：' + str(today_pred)
        return today_pred

    def LinearRegession(self, code):
        self.dodata(code)
        # 构造模型
        model = LogisticRegression()
        # 训练模型
        model.fit(self.x_train, self.y_train)
        # 模型预测
        self.preds = model.predict(self.x_test)
        # 今天的收盘价预测
        today_pred = round(self.preds[self.preds.shape[0] - 1], 2)
        today_pred = 'Line预测价格：' + str(today_pred)
        return today_pred

    # 统计信息方法
    def get_stats(self, s):
        # del Nan data
        s = s.dropna()
        # 盈利次数：收益率大于0出现的总数量
        wins = len(s[s > 0])
        # 亏本次数：收益率小于0出现的总数量
        losses = len(s[s < 0])
        # 盈亏平衡次数
        evens = len(s[s == 0])
        # 盈利平均值
        mean_w = round(s[s > 0].mean(), 2)
        # 亏损平均值
        mean_l = round(s[s < 0].mean(), 2)
        # 盈利与亏损比例：盈利次数/亏损次数
        win_r = round(wins / losses, 2)
        # 平均收益
        mean_trd = round(s.mean(), 2)
        # 标准值：收益率的标准差
        sd = round(np.std(s), 2)
        # 最大亏损：收益率最小值
        max_l = round(s.min(), 3)
        # 最大盈利
        max_w = round(s.max(), 2)
        # 夏普比率：（收益率平均值-无风险利率）/收益率标准差
        sharpe_r = round((s.mean() - 0.05) / np.std(s), 2)
        # 交易次数
        cnt = len(s)
        stats_res = f'交易次数：{str(cnt)},<br>盈利次数：{str(wins)},<br>亏损次数：{str(losses)},<br>盈亏平衡次数：{str(evens)},<br>盈利平均值：{str(mean_w)},<br>亏损平均值：{str(mean_l)},<br>盈利与亏损比例：{str(win_r)},<br>平均收益：{str(mean_trd)}<br>标准差：{str(sd)},<br>最大亏损：{str(max_l)},<br>最大盈利：{str(max_w)},<br>夏普比率：{str(sharpe_r)},'
        return stats_res

    def daily_stats(self):
        # 日内交易
        # 日内交易收益率
        self.spy['Daily Change Rate'] = pd.Series((self.spy['close'] - self.spy['open']) / self.spy['open']) * 100
        return self.get_stats(self.spy['Daily Change Rate'])

    def id_stats(self):
        # 日间交易
        # 日间交易收益率
        self.spy['id Change Rate'] = pd.Series(
            (self.spy['close'] - self.spy['open'].shift(1)) / self.spy['clsoe']) * 100
        return self.get_stats(self.spy['id Change Rate'])

    def overnight_stats(self):
        # 隔夜交易
        # 隔夜交易收益率
        self.spy['overnight Change Rate'] = pd.Series(
            (self.spy['open'] - self.spy['close'].shift(1)) / self.spy['close'].shift(1)) * 100
        return self.get_stats(self.spy['overnight Change Rate'])

    def custom_stats(self):
        # self.y_test:实际收盘价，self.preds:预测收盘价，总共20%测试数据
        tf = pd.DataFrame(list(zip(self.y_test, self.preds)), columns=['Next Day Close', 'Predicted Next Close'],
                          index=self.y_test.index)
        # 当前日收盘价（不存在20个nan），包含20%的测试数据
        cdc = self.sp20[['close']]
        # 下一日开盘价（存在20个nan），包含20%的测试数据
        ndo = self.spy[['open']].shift(-1)
        ccc = pd.merge(tf, cdc, left_index=True, right_index=True)
        ddd = pd.merge(ccc, ndo, left_index=True, right_index=True)  # 明日实收，明日预测收，当前日实收，Ingrid开盘价（数据行数不一致，不可用concat）
        tf2 = ddd.assign(Signal=ddd.apply(self.get_signal, axis=1))  # assign是直接向dataframe对象添加新的一列：是否交易
        tf3 = tf2.assign(PnL=tf2.apply(self.get_ret, axis=1))  # assign是直接向dataframe对象添加新的一列：收益率
        # 明日实收：next day close
        # 明日预测收：predicted next close
        # 当前日实收：close
        # 明日开盘价：open
        # 是否交易：Signal
        # 收益率：PnL
        return self.get_stats(tf3['PnL'])

    # 自定义新的量化交易策略
    def get_signal(self, r):
        if (r['open'] < r['Predicted Next Close']):
            return 1
        else:
            return 0

    # 收益率
    def get_ret(self, r):
        if r['Signal'] == 1:
            return ((r['Next Day Close'] - r['open']) / r['open']) * 100
        else:
            return 0


# q = Quant()
# print(q.get_realtime_stock_info('000001'))
# print(q.catch_data('000019','2019-01-01','2019-12-04'))
# print(q.out_csv('000019'))
