import numpy as np
from tensorflow.keras.datasets import fashion_mnist

(train_input, train_target) , (test_input, test_target) = fashion_mnist.load_data()

print(len(train_input),train_input.shape) # 60000
print(len(test_input),test_input.shape)
print(train_input[0])
# print(test_input[0].shape, len(test_input))  # (28,28) , 길이 10000
print('target label 체크 : ')
print(np.unique(train_target, return_counts=True)) # return_counts=True : 중복데이터의 개수까지 
print(train_target[0]) # label 9

# import matplotlib.pyplot as plt
# #print(train_input[0])
# plt.imshow(train_input[0],cmap='gray')  # label 9인  input 이미지 체크
# plt.show()
#훈련데이터 스케일 0~255 사이의 데이터를 0~1 사이로 변환
#합성곱 층은 3차원 데이터 입력 기대
#print(train_input.shape)
train_scaled = train_input.reshape(-1,28,28,1) / 255.0
print(train_scaled.shape)

from sklearn.model_selection import train_test_split
#  훈련 세트와 검증 세트로 분할
train_scaled, val_scaled, train_target , val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)
print(len(train_scaled)) # 48000
print(len(val_scaled)) # 12000

# #print(val_scaled.shape)
# #print(len(val_scaled))
# #print(val_scaled[0:1])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#### 합성곱(CNN) 신경망 생성 ####
model = Sequential()

# 32개 필터, (3,3) 커널 사이즈, relu 활성화함수,  same패딩,  (28,28,1) 크기 input
# 합성곱 층
model.add( Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=(28,28,1) ) )

# 폴링 층 추가, (2,2) 최대값 풀링
model.add(MaxPooling2D(2))
# print(model.summary())

# 합성곱 층 추가
model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))

#  풀링 층 추가
model.add(MaxPooling2D(2))

# 펼치기 ( Fatten() )
model.add(Flatten())
#print(model.summary())
# # 밀집층 추가
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.4))

#출력층 , 소프트맥스 활성화 함수
model.add(Dense(10,activation='softmax'))

# model.summary()
# from tensorflow.keras.utils import plot_model
# plot_model(model)
# plot_model(model, show_shapes=True, to_file='cnn-architecture.png', dpi=300)


## 모델 컴파일과 훈련 ##

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

checkpoint_cb = ModelCheckpoint('best-cnn-model.h5')
early_stopping_cb = EarlyStopping(patience=3, restore_best_weights=True)

# 모델 학습
history = model.fit(train_scaled, train_target, epochs=100, validation_data=(val_scaled, val_target),callbacks=[checkpoint_cb, early_stopping_cb])

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

# 평가
#print('\n Val Accuracy : %.4f' %model.evaluate(val_scaled, val_target, verbose=0)[1] )

# plt.imshow(val_scaled[0].reshape(28,28), cmap='gray_r')
# plt.show()


#예측
# preds = model.predict(val_scaled[0:1])
# print(preds)

# 예측 2
# import cv2
# import numpy as np
#
# img_path = 'bag.jpg'
# img = cv2.imread(img_path,0)
#
# print(img)
# img_reverted = cv2.bitwise_not(img)
#
# new_img = img_reverted / 255.0
# print(new_img)
# new_img = new_img.reshape(1,28,28,1)
# preds = model.predict(new_img[0:1])
# print(preds)

# plt.bar(range(1,11), preds[0])
# plt.xlabel('class')
# plt.ylabel('prob.')
# plt.show()

# classes = ['티셔츠', '바지', '스웨터', '드레스','코트', '샌달', '셔츠', '스니커즈', '가방', '앵클 부츠']
#
# import numpy as np
#
# # preds 가장 큰 인덱스 찾아 리스트 인덱스 전달
# print(classes[np.argmax(preds)])
#
# #test 셋 데이터 평가
# test_scaled = test_input.reshape(-1, 28, 28, 1) / 255.0
#
# print('\n Tes Accuracy : %.4f' %(model.evaluate(test_scaled, test_target,verbose=0)[1]) )
