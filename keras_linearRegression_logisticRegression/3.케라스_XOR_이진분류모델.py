# tensorflow Gpu 사용 메시지 미출력하게 하는 방법 : 아래 2줄 추가
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

model = Sequential()
model.add(Dense(6,input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_s = 1
epoch_c = 1000
model.fit(x,y,batch_size=batch_s, epochs=epoch_c, verbose=1 )
print( model.evaluate(x,y,batch_size=batch_s) )

pre = model.predict(np.array([[1,1],[0,1],[1,0]]))

print(pre)
pre = np.where(pre > 0.5, 1, 0 )
# np.where( 조건식, 조건식 True일 경우 반환값, 조건식 False일 경우 반환값)
# 두 번째와 세 번째 인수를 생략할 경우 조건식과 일치하는 요소의 인덱스 반환
print(pre)