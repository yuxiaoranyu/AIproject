import os
import requests
from parsel import Selector
from fake_useragent import UserAgent
import time
import pymysql
import csv
import pandas as pd
import re


class House(object):
    def __init__(self):
        #构造请求头
        ua=UserAgent()
        self.headers={'user-agent':ua.Chrome}
        #数据保存路径
        self.PATH=os.path.abspath(os.path.dirname(__file__))+os.sep+'data'+os.sep
        #暂存数据的列表
        self.data_list=[]
    def diedai(self,lis):
        for i in lis:
            print(i)

    def start_spider(self,title,page):
        #爬取数据：http://gz.lianjia.com/ershoufang/{title}/pg{page}
        url=f'http://gz.lianjia.com/ershoufang/{title}/pg{page}'
        print(url)
        wr=requests.get(url,headers=self.headers,stream=True)
        sel=Selector(wr.text)

        #获取房屋总价
        price_t=sel.xpath('//div[@class="totalPrice"]//text()').extract()
        totalPrices=[i for i in price_t if i !='万']
        # print(totalPrices)
        #获取房屋单价
        unitPrices=sel.xpath('//div[@class="unitPrice"]/span//text()').extract()
        # print(unitPrices)
        #获取标题
        ddecs=sel.xpath('//div[@class="title"]/a//text()').extract()
        # print(ddecs)
        #获取楼盘名称
        lo=sel.xpath('//div[@class="positionInfo"]/a//text()').extract()
        # print(lo)
        loupans=[]
        for i in range(0,len(lo)-1,2):
            loupans.append('-'.join(lo[i:i+2]))
        #先抓取全部的houseInfo的信息，然后再逐条提取
        houseInfo=sel.xpath('//div[@class="houseInfo"]//text()').extract()
        house_types,areas,towards,renovations,positionInfos=[],[],[],[],[]
        for i in houseInfo:
            #对信息进行分割
            info=i.split('|')
            #房子类型
            house_type=info[0].strip()
            house_types.append(house_type)
            #面积
            area=info[1].strip()
            areas.append(area)
            #朝向
            toward=info[2].strip()
            towards.append(toward)
            #装修类型
            renovation=info[3].strip()
            renovations.append(renovation)
            #位置信息
            positionInfo=info[4].strip()
            positionInfos.append(positionInfo)

        for ddec,loupan,area,house_type,unitPrice,toward,renovation,positionInfo,totalPrice in zip(ddecs,loupans,areas,house_types,unitPrices,towards,renovations,positionInfos,totalPrices):
            data_dict={}
            data_dict['title']=title
            data_dict['ddec']=ddec
            data_dict['loupan']=loupan
            data_dict['area']=area
            data_dict['house_type']=house_type
            data_dict['unitPrice']=unitPrice
            data_dict['toward']=toward
            data_dict['renovation']=renovation
            data_dict['positionInfo']=positionInfo
            data_dict['totalPrice']=totalPrice
            self.data_list.append(data_dict)
            print(data_dict)

    def get_house(self,fresh=False):
        # 爬取数据存入数据库
        if fresh:
            try:
                os.remove(self.PATH+'gz_house.csv')
            except Exception as e:
                print(f'文件不存在：{e}')
        if not os.path.exists(self.PATH+'gz_house.csv'):
            titles=['tianhe','yuexiu','liwan','haizhu','panyu','baiyun','huangpu','conghua','zengcheng','huadu','nansha','nanhai','shunde']
            pages=list(range(1,11))
            for page in pages:
                for title in titles:
                    self.start_spider(title,page)
                    time.sleep(10)
            #连接数据库
            db=pymysql.Connect(
                host='localhost',
                port=3306,
                user='root',
                passwd='root',
                db='lianjia',
                charset='utf8'
            )
            cursor=db.cursor()
            try:
                sql2='create table if not exists house(id int NOT NULL AUTO_INCREMENT PRIMARY KEY ,title varchar (255) DEFAULT NULL ,loupan varchar (255) DEFAULT NULL ,ddec varchar (255) default null ,house_type varchar (255) default null ,area varchar (255) default null ,toward varchar (255) default null ,positionInfo varchar (255) default null ,unitPrice varchar (255) default null ,totalPrice varchar (255) default null ,renovation varchar (255) default null ) default charset=utf8;'
                cursor.execute(sql2)
            except Exception as e:
                print(f'数据库创建表已存在异常：{e}')
            db.commit()

            if fresh:
                #清空house表中的数据
                sql='truncate table house;'
                cursor.execute(sql)
                db.commit()

            for data_dict in self.data_list:
                title=data_dict['title']
                area=data_dict['area']
                house_type=data_dict['house_type']
                unitPrice=data_dict['unitPrice']
                renovation=data_dict['renovation']
                positionInfo=data_dict['positionInfo']
                totalPrice=data_dict['totalPrice']
                toward=data_dict['toward']
                ddec=data_dict['ddec']
                loupan=data_dict['loupan']
                sql = sql = f"insert into house(title,loupan,ddec,house_type,area,toward,positionInfo,totalPrice,unitPrice,renovation) values('{title}','{loupan}','{ddec}','{house_type}','{area}','{toward}','{positionInfo}','{totalPrice}','{unitPrice}','{renovation}')"
                #args=(title,loupan,ddec,house_type,area,toward,positionInfo,totalPrice,unitPrice,renovation)
                cursor.execute(sql)
                db.commit()
            cursor.close()
            db.close()

    def get_count(self):
        #统计数量
        db = pymysql.Connect(
            host='localhost',
            port=3306,
            user='root',
            passwd='root',
            db='lianjia',
            charset='utf8'
        )
        cursor=db.cursor()
        sql='select count(1) from house;'
        cursor.execute(sql)
        total_count=cursor.fetchall()[0]
        cursor.close()
        db.close()
        print(f'数据量为：{total_count}条')
        return total_count

    def out_csv(self,fresh=False):
        if fresh:
            try:
                os.remove(self.PATH+'gz_house.csv')
            except Exception as e:
                print(f'文件不存在：{e}')
        res={}
        if not os.path.exists(self.PATH+'gz_house.csv'):
            config={'host':'localhost',
                    'port':3306,
                    'user':'root',
                    'passwd':'root',
                    'db':'lianjia',
                    'charset':'utf8',
                    'cursorclass':pymysql.cursors.DictCursor}
            db=pymysql.Connect(**config)
            cursors=db.cursor()
            sql='select * from house;'
            try:
                cursors.execute(sql)
                results=cursors.fetchall()
                data_list=[]
                for row in results:
                    #将数据写入csv文件
                    data_dict={}
                    data_dict['title']=str(row['title'])
                    data_dict['ddec']=str(row['ddec'])
                    data_dict['loupan'] = str(row['loupan'])
                    data_dict['area'] = str(row['area'])
                    data_dict['house_type'] = str(row['house_type'])
                    data_dict['unitPrice'] = str(row['unitPrice'])
                    data_dict['toward'] = str(row['toward'])
                    data_dict['renovation'] = str(row['renovation'])
                    data_dict['positionInfo'] = str(row['positionInfo'])
                    data_dict['totalPrice'] = str(row['totalPrice'])
                    data_list.append(data_dict)
                with open(self.PATH+'gz_house.csv','w',encoding='utf-8') as f:
                    #表头
                    title=data_list[0].keys()
                    #创建一个writer对象
                    writer=csv.DictWriter(f,title)
                    #写入表头
                    writer.writeheader()
                    #批量写入数据
                    writer.writerows(data_list)
                res['msg']="导出数据正常：{self.PATH+'gz_house.csv'}"
            except Exception as e:
                res['msg']='导出数据正常！'
            finally:
                cursors.close()
                db.close()
        else:#数据已经保存到本地
            res['msg']="导出数据正常：{self.PATH+'gz_house.csv'}"
        return res

    def do_data(self,fresh=False):
        #清洗数据，数据离散化
        preprocess_path=self.PATH+'preprocess_gz_house.csv'#数据处理后的路径
        if fresh:
            try:
                os.remove(self.PATH+'gz_house.csv')
            except Exception as e:
                print('文件不存在：{}'.format(e))
        if not os.path.exists(preprocess_path):
            try:
                file_path=self.PATH+'gz_house.csv'#数据处理后保存的路径
                df=pd.read_csv(file_path)
                df['gz_totalPrice']=df['totalPrice']
                df.loc[:,'gz_unitPrice']=df.loc[:,'unitPrice'].apply(self.parse_info_math)
                df['gz_title']=df['title']
                df.loc[:,'gz_area']=df.loc[:,'area'].apply(self.parse_info_area)
                bed_and_room=df['house_type'].apply(self.parse_info_house_type)
                df=df.join(bed_and_room)
                df.loc[:,'gz_toward']=df.loc[:,'toward'].apply(self.parse_info_toward)
                df.loc[:,'gz_renovation']=df.loc[:,'renovation'].replace(['精装','简装'],[1,0])
                df.loc[:,'gz_positionInfo']=df.loc[:,'positionInfo'].apply(self.parse_info_math)
                new_df=df[['gz_totalPrice','gz_unitPrice','gz_title','gz_area','gz_beds','gz_rooms','gz_toward','gz_renovation','gz_positionInfo']]
                # print(df.head())
                new_df.to_csv(preprocess_path)
                print('文件初次处理成功')
            except Exception as e:
                print(e)
                return e
        else:
            result=f'文件已处理成功，无需再处理：{preprocess_path}'

    def parse_info_area(self,row):
        return row.strip('平米')

    def parse_info_house_type(self,row):
        #方法一
        # beds=row.split('室')[0]
        # rooms=row.split('室')[1].strip('厅')
        bed_rooms=re.sub('\D','',row)#去除非数字
        if len(bed_rooms)==0:
            beds=0
            rooms=0
        else:
            beds=bed_rooms[0]
            rooms=bed_rooms[1]
        return pd.Series({'gz_beds':beds,"gz_rooms":rooms})

    def parse_info_math(self,row):
        return re.sub('\D','',row)

    def parse_info_toward(self,row):
        return str(row)[0]

c=House()
# c.start_spider('tianhe',1)
# c.get_house()
# c.get_count()
# c.out_csv()
c.do_data()