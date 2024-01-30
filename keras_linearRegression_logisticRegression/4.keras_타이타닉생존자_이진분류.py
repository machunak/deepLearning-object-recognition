# tensorflow Gpu 사용 메시지 미출력하게 하는 방법 : 아래 2줄 추가
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import pandas as pd

pd.set_option('display.max_rows',20)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width',1000)

titanic_df = pd.read_csv('13.케라스_선형회귀_로지스틱회귀/titanic_passengers.csv')
print(titanic_df.shape)
print(titanic_df.head())

# 분석에 사용할 특징 데이터 셋 선택
# Sex, Age, Pclass 컬럼 데이터셋이 생존에 영향을 주는걸로 가설

# Sex 컬럼 'female', 'male' 문자열 데이터를  여성 : 1, 남성 : 0 으로 변경
titanic_df['Sex'] = titanic_df['Sex'].map({'female':1, 'male':0})

# 결측치 검사
print(titanic_df.info())
print(titanic_df.loc[titanic_df['Age'].isnull(),['Age']])  # 177개 NAN

# 결측치 채우기 : 평균 값
titanic_df['Age'].fillna(value=titanic_df['Age'].mean(), inplace=True)
#titanic_df['Age'].fillna(value=0, inplace=True)
#titanic_df.dropna(how='any', subset='Age', inplace=True)
print(titanic_df.info())
#print(titanic_df.head(20))

# PClass ==> 1등석, 2등석, 3등석 구분
onehot_Pclass = pd.get_dummies(titanic_df['Pclass'], prefix='Class')
print(onehot_Pclass)  # Class_1  Class_2  Class_3 컬럼 생성

# titanic_df 와 onehot_Pclass 와 결합
titanic_df = pd.concat([titanic_df,onehot_Pclass], axis=1)
print(titanic_df.head(10))

# 데이터셋 준비
# 입력 데이터 셋 : [Sex, Age, Class_1 ,  Class_2 ] 컬럼
# 타깃 데이터 셋 : [Survived] 컬럼

titanic_Info = titanic_df[['Sex','Age','Class_1','Class_2']]
titanic_survival = titanic_df['Survived']
# print(titanic_Info.head(5))
# print(titanic_survival.head(5))


# train dataset / test dastaset : 훈련,테스트 데이터셋 분리
from sklearn.model_selection import train_test_split
train_input,test_input,train_target,test_target = \
    train_test_split(titanic_Info, titanic_survival, random_state=42)

print(train_input.shape)
# 스케일 변환
# StandardScaler : 모든 값이 평균 0, 표준편차가 1인 정규분포로 변환
# MinMaxScaler : 최소값 0, 최대값 1로 변환
# RobustScaler : 중앙값 과 IQR(interquartile range): 25%~75% 사이의 범위 사용해 변환
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_input)
# fit() 했음으로 transform만 하면됨
test_scaled = scaler.transform(test_input)
print(train_scaled) # numpy.ndarray 변환 출력
print(train_scaled[:,0].std()) # 0번째 열(Sex) 데이터 표준편차 1

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(input_shape=(4,), units=8, activation='relu'))
model.add(Dense(units=4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='sgd', loss='binary_crossentropy',
                 metrics=['accuracy'])
model.summary()
model.fit(train_scaled, train_target, batch_size=64, epochs=1000, verbose=1)

score = model.evaluate(test_scaled, test_target)
print('Test acc : ', score[1])

model.summary()

print( model.layers[0].weights )
