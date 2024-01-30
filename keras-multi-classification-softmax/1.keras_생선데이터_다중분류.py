# tensorflow Gpu 사용 메시지 미출력하게 하는 방법 : 아래 2줄 추가
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import pandas as pd

# 과학적 표기법 대신 소수점 6자리까지 나타낸다.
np.set_printoptions(precision=8, suppress=True)

fish_data = pd.read_csv('fish_data.csv')
print(fish_data)
fish_input = fish_data[['Weight','Length','Diagonal','Height','Width']].to_numpy()
print(fish_input[:5])
fish_target = fish_data['Species'].to_numpy()
print(fish_data['Species'].unique())

print(fish_target)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y_encoded = le.fit_transform(fish_target)
print('Y_encoded : ', Y_encoded)
print('le.clasess_ : ', le.classes_)

from tensorflow.keras.utils import to_categorical
Y_onehot = to_categorical(Y_encoded)
print(Y_onehot)  # 원핫인코딩 변환

# 훈련데이터 / 테스트데이터 셋 분리
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = \
    train_test_split(fish_input, Y_onehot, random_state=42)

print(train_input.shape)
print(test_input.shape)

#StandardScaler 클래스 활용 표준화 전처리
from sklearn.preprocessing import StandardScaler
Scaler = StandardScaler()
Scaler.fit(train_input)
train_scaled = Scaler.transform(train_input)
test_scaled = Scaler.transform(test_input)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

lr_model = Sequential()
lr_model.add(Dense(input_shape=(5,), units=7, activation='softmax'))
lr_model.summary()
lr_model.compile(optimizer='sgd', loss='categorical_crossentropy',
                 metrics=['accuracy'])
lr_model.fit(train_scaled, train_target, epochs=1000, verbose=1)

score = lr_model.evaluate(test_scaled, test_target)
print('Test acc : ', score[1])

#lr_model.summary()

print( lr_model.layers[0].weights )

print(test_target[0])
print(le.classes_[np.argmax(test_target[0])])
pre = lr_model.predict(test_scaled[0:1])
print('pre : ', pre)
print(le.classes_[np.argmax(pre)])