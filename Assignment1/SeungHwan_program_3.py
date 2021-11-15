import pandas as pd

lottery = pd.read_csv('lottery.csv')

#세번째 숫자와 네번째 숫자의 평균 열 추가
lottery['th_fo'] = (lottery.third + lottery.fourth)/2

import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

#th_fo와 bonus를 데이터 변수에 선언
data_points = lottery[['th_fo', 'bonus']]

#클러스터 갯수 정하기
kmeans = KMeans(n_clusters = 6).fit(data_points)

#로터리 데이터프레임에 클러스터 열 추가하기
lottery['cluster_id'] = kmeans.labels_

#시각화
sns.lmplot('th_fo', 'bonus', data = lottery, fit_reg = False, scatter_kws = {"s" : 10}, hue = "cluster_id")

#라벨 붙이기
plt.title('Relationship Between th_fo_mean and bonus')

plt.xlabel('Th_Fo_Mean = (Third_Num + Foruth_Num)/2')

plt.ylabel('Bonus_Num')

'''
#클러스터 갯수별로 몇개씩 나뉘는지 확인하는 과정
cnt = lottery['cluster_id'].value_counts()

print(cnt)
'''

plt.show()
