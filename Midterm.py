#!/usr/bin/env python
# coding: utf-8

# In[241]:


#Task1
import pandas as pd

trade = pd.read_csv('C:/Users/dg798/Desktop/21년 3학년 2학기/인공지능/takehome/ite351-takehome-midterm/2018-06-trade.csv')

nu_li = [] # 범위 안에 들어오는 번호 리스트

ex_pf = 0 # 해당 번호까지의 이익

qnt = 0 # quantity 값 변수

qnt_li = [] # 범위 안에 들어오는 accumulative quantity 리스트

ex_pf_li = [] # 범위 안에 들어오는 exact profit 리스트

trade.quantity = trade.quantity.round(4)

print(trade)

for i in range (1089) :
    
    #sell일 경우
    if trade['side'][i]  == 1 :
        
        qnt += trade['quantity'][i]
        
        ex_pf += (trade['quantity'][i] * trade['price'][i])
    
    #buy일 경우
    if trade['side'][i]  == 0 :
        
        qnt -= trade['quantity'][i]
        
        ex_pf -= (trade['quantity'][i] * trade['price'][i])
    
    #quantity 범위 설정
    if qnt > -10 and qnt < 10 :
        
        qnt_li.append(qnt)
        
        nu_li.append(i)
        
        ex_pf_li.append(ex_pf)

#번호, 축적된 양, 정확한 이익 데이터를 뽑아내 새로운 데이터프레임 생성
data = {
    "point" : nu_li,
    "acc_qnt_value" : qnt_li,
    "exact_profit" : ex_pf_li
}

columns = ["point", "acc_qnt_value", "exact_profit"]
result = pd.DataFrame(data, columns = columns)
        
print(result)

#범위 내의 profit 합계
print("exact_profit is ", result.exact_profit.sum())


# In[120]:


#Task2
import pandas as pd

df = pd.read_csv('C:/Users/dg798/Desktop/21년 3학년 2학기/인공지능/takehome/ite351-takehome-midterm/2018-06-trade.csv')

total = pd.DataFrame()

#timestamp, side 열 추출
total['timestamp'] = df.timestamp
total['timestamp'] = pd.to_datetime(total['timestamp']) #object 자료형 timestamp로 변환
#total.info()로 열 자료형 확인 가능

total['side'] = df.side

#side 값에 따라 새로운 데이터프레임 생성
sell = total.loc[total.side == 1]
buy = total.loc[total.side == 0]

#side를 모두 1로 변경하여 갯수 파악
total.side = 1
buy.side = 1

#resample 과정
total = total.resample('1H', on ='timestamp').sum()
sell = sell.resample('1H', on ='timestamp').sum()
buy = buy.resample('1H', on ='timestamp').sum()

#merge 전 side 이름 구분
total.rename(columns={'side':'Total_Transaction'}, inplace = True)
sell.rename(columns={'side':'Sell_Transaction'}, inplace = True)
buy.rename(columns={'side':'Buy_Transcation'}, inplace = True)

#dataframe 3개 merge
sell_buy = pd.merge(sell, buy, on = 'timestamp')
total = pd.merge(total, sell_buy, on = 'timestamp')

#그래프 그리기
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt

total.plot(marker='o', figsize=(20,5))
plt.title("Hourly Transaction Count", fontsize = 30)
plt.xlabel("Timestamp_Hour")
plt.grid(True) #눈금그리기
mpl.rc('axes', labelsize = 15)
mpl.rc('xtick', labelsize = 15)
mpl.rc('ytick', labelsize = 15)
mpl.rc('legend', fontsize = 15)


# In[237]:


#Task3
import pandas as pd

trade = pd.read_csv('C:/Users/dg798/Desktop/21년 3학년 2학기/인공지능/takehome/ite351-takehome-midterm/2018-06-trade.csv')


order_01 = pd.read_csv('C:/Users/dg798/Desktop/21년 3학년 2학기/인공지능/takehome/ite351-takehome-midterm/2018-06-01-orderbook.csv')
order_02 = pd.read_csv('C:/Users/dg798/Desktop/21년 3학년 2학기/인공지능/takehome/ite351-takehome-midterm/2018-06-02-orderbook.csv')

#시계열 자료 변환
def to_datetime (df) :
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df

trade = to_datetime(trade)
order_01 = to_datetime(order_01)
order_02 = to_datetime(order_02)

#MidPrice 계산하기 위한 quantity분리작업
def Calculate_MidPrice (df) :
    
    mid = pd.DataFrame()
    mid['price'] = df['price']
    mid['type'] = df['type']
    mid['timestamp'] = df['timestamp']
    #Ask와 Bid 데이터프레임 구분
    Ask = mid.loc[mid.type == 1]
    Bid = mid.loc[mid.type == 0] 
    #resample
    Ask = Ask.resample('1s', on ='timestamp').min()
    Bid = Bid.resample('1s', on ='timestamp').min()
    #timestamp 중복 제거 및 MidPrice 계산을 위한 merge작업
    del Ask['timestamp']
    del Bid['timestamp']
    Merge = pd.merge(Ask, Bid, on = 'timestamp')
    Merge['mid_price'] = (Merge['price_x']+Merge['price_y'])/2
    #MidPrice만 도출
    del Merge['price_x']
    del Merge['type_x']
    del Merge['price_y']
    del Merge['type_y']
    
    return Merge

order_01_mid = Calculate_MidPrice(order_01)
order_02_mid = Calculate_MidPrice(order_02)


#quantity average 구하기
def quantity_avg (df) :
    #quantity, type, timestamp 데이터프레임 새로 만들기
    qnt = pd.DataFrame()
    qnt['quantity'] = df['quantity']
    qnt['type'] = df['type']
    qnt['timestamp'] = df['timestamp']
    #Ask와 Bid 데이터프레임 구분
    Ask = qnt.loc[qnt.type == 1]
    Bid = qnt.loc[qnt.type == 0]
    #resample
    Ask = Ask.resample('1s', on ='timestamp').mean()
    Bid = Bid.resample('1s', on ='timestamp').mean()
    #Merge
    Merge = pd.merge(Ask, Bid, on = 'timestamp')
    del Merge['type_x']
    del Merge['type_y']
    Merge.rename(columns = {'quantity_x' : 'askQty', 'quantity_y' : 'bidQty'}, inplace = True)
    
    return Merge

order_01_qnt = quantity_avg(order_01)
order_02_qnt = quantity_avg(order_02)


#price average 구하기
def price_avg (df) :
    #price, type, timestamp 데이터프레임 새로 만들기
    price = pd.DataFrame()
    price['price'] = df['price']
    price['type'] = df['type']
    price['timestamp'] = df['timestamp']
    #Ask와 Bid 데이터프레임 구분
    Ask = price.loc[price.type == 1]
    Bid = price.loc[price.type == 0]
    #resample
    Ask = Ask.resample('1s', on ='timestamp').mean()
    Bid = Bid.resample('1s', on ='timestamp').mean()
    #Merge
    Merge = pd.merge(Ask, Bid, on = 'timestamp')
    del Merge['type_x']
    del Merge['type_y']
    Merge.rename(columns = {'price_x' : 'askPx', 'price_y' : 'bidPx'}, inplace = True)

    return Merge

order_01_price = price_avg(order_01)
order_02_price = price_avg(order_02)


#book price, Bfeature, Alpha 계산
def p_B_A_cal (order_qnt, order_price, order_mid) :
    
    df = pd.merge(order_qnt, order_price, on = 'timestamp')
    df = pd.merge(df, order_mid, on = 'timestamp')
    df['book_price'] = (((df.askQty*df.bidPx)/df.bidQty) + ((df.bidQty*df.askPx)/df.askQty)) / (df.bidQty+df.askQty)
    df['Bfeature'] = df.book_price - df.mid_price
    df['Alpha'] = df.Bfeature * df.mid_price * 0.004
    #midprice, bfeature, alpha 열만 남기기
    del df['askQty']
    del df['bidQty']
    del df['askPx']
    del df['bidPx']
    del df['book_price']
    
    return df


result_01 = p_B_A_cal (order_01_qnt, order_01_price, order_01_mid)
result_02 = p_B_A_cal (order_02_qnt, order_02_price, order_02_mid)


#trade파일 열 제거
del trade['quantity']
del trade['fee']
del trade['amount']

#order 1,2 파일 concat
result = pd.concat([result_01, result_02])

#trade파일과 order1,2파일 merge
trade_result = pd.merge(trade, result, on = 'timestamp')


#최종 trade 파일
print(trade_result)

#csv파일로 내보내기
trade_result.to_csv('C:/Users/dg798/Desktop/21년 3학년 2학기/인공지능/takehome/ite351-takehome-midterm/2018-06-trade-modify.csv')


# In[59]:


#Task4
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from matplotlib.colors import ListedColormap

dataset = pd.read_csv('C:/Users/dg798/Desktop/21년 3학년 2학기/인공지능/takehome/ite351-takehome-midterm/2018-06-trade-modify.csv')


#선택할 열 제외하고 삭제
#timestamp는 randomforest할 때 데이터 값이 다르므로 제외
del dataset['timestamp']
#price값이 거래의 실제 가격이므로 mid_price는 중요도가 떨어진다고 생각해서 제외
del dataset['mid_price']
#Alpha는 Bfeature와 밀접한 연관이 있으므로 제외함
del dataset['Alpha']
dataset = dataset[['price', 'Bfeature', 'side']]

#비어있는 값 채우기 - 앞의 열로 설정
dataset['Bfeature'][986] = dataset['Bfeature'][985]

# target을 side로 설정
X = dataset.iloc[:, [0, 1]].values
y = dataset.iloc[:, 2].values

#9:1비율로 설정
#test_size 값 비교
#8:2 -> 83%
#75:25 -> 83%
#70:30 -> 77%
#60:40 -> 79%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

#price와 Bfeature 범위 간략화
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test) 

#정확도 확인
cm = confusion_matrix(y_test, y_pred)

#정확도
print ("Accuracy : ", accuracy_score(y_test, y_pred))

#실제 값과 예측 값 비교
df = pd.DataFrame({'Real Values':y_test, 'Predicted Values':y_pred})
print(df)

#그래프 시각화
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('RandomForestClassification')
plt.xlabel('price')
plt.ylabel('Bfeature')
plt.legend()

