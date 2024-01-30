# tensorflow Gpu 사용 메시지 미출력하게 하는 방법 : 아래 2줄 추가
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import pandas as pd

Houseinfo = pd.read_csv('13.케라스_선형회귀_로지스틱회귀/BostonHousing.csv')
print(Houseinfo)
print(Houseinfo.info())

x = Houseinfo.iloc[:,0:13]
y = Houseinfo.iloc[:,13]


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=42)

# 특성데이터 크기가 상이할 경우 정규화 중요!!
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_x)
# fit() 했음으로 transform만 하면됨
test_scaled = scaler.transform(test_x)

model = Sequential()
model.add(Dense(30, input_dim=13, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1,activation='linear')) # activation = 'linear' 생략 가능 , default = 'linear'임

model.compile(loss='mse', optimizer='adam',metrics=['mse'])
# 회귀모델 경우 성능평가 모니터링 지표로 평균제곱오차 사용
# 결국 loss 와 동일 함으로 loss만 모니터링 해두 됨!
model.fit(train_scaled, train_y, epochs=200, batch_size=10, verbose=1)

pre = model.predict(test_scaled).flatten() # 예측값을 1차원 배열로 펼침

for i in range(10):  # 10개 값만 비교
    print('실제가격 : {:.3f}, 예상가격 : {:.3f}'.format(test_y.iloc[i], pre[i]))
