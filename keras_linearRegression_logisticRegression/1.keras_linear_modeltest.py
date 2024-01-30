# tensorflow Gpu 사용 메시지 미출력하게 하는 방법 : 아래 2줄 추가
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(0, 10, 10)
print(X.shape, type(X.shape))
#
# 함수 호출시 * 및 ** 연산자를 사용하여 인수 모음을 각각 별도의 위치 또는 키워드 인수로
# unpack할 수 있음
Y = X + np.random.randn(*X.shape)  # *(10,) ==> 10 으로 unpack
print(Y)

model = Sequential()  # keras 신경망 모델 (순차적으로 층을 추가(쌓는)하는 모델), 순차모델
# input_dim : 입력 뉴런의 수 설정
# units : 출력(히든) 뉴런 수 설정
# 활성화함수 설정
# activation='linear'  :  선형회귀, 입력 뉴련과 가중치로 계산된 결과값이 그대로 출력
# activation='relu'  :  rectifier 함수, 주로 은익층에 사용
# activation='sigmoid'  :  시그모이드 함수, 이진 분류 문제의 출력층에 주로 사용
# activation='softmax'  :  소프트맥스 함수, 다중 분류 문제의 출력층에 주로 사용
# Dense() : 은닉층 추가
model.add(Dense(input_dim=1, units=1, activation='linear',use_bias=False))
# 옵티마이저 : 가중치 최적화 방법 설정 ( sgd, adam , rmsprop 등)
sgd = optimizers.SGD(learning_rate=0.01) # 확률적 경사하강, 학습률 0.01
# 모델 구성후 compile()메서드 호출해서 모델 학습 과정을 설정, 모델 실행(훈련) 전 필요한 설정 결합
model.compile(optimizer=sgd, loss='mse') # 옵티마이저(최적화) : 확률적경사하강, 손실함수 : 평균제곱오차

weights = model.layers[0].get_weights()
#print(weights[0][0][0])
w = weights[0][0][0]
print('w begin fit() : ', w)

model.fit(X, Y, batch_size=10, epochs=10, verbose=1)

weights = model.layers[0].get_weights()  # 훈련 이후 수정된 w 계수 확인
w = weights[0][0][0]
print('w after fit() : ', w)

import matplotlib.pyplot as plt
plt.plot(X,Y, label='data')
plt.plot(X, w*X, label = 'prediction')
plt.legend()
plt.show()