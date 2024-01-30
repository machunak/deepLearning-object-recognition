import numpy as np
from tensorflow.keras.datasets import fashion_mnist

(train_input, train_target) , (test_input, test_target) = fashion_mnist.load_data()

print(len(train_input),train_input.shape) # 60000
print(test_input[0].shape, len(test_input))  # (28,28) , 길이 10000
print('target label 체크 : ')
print(np.unique(train_target, return_counts=True))
print(train_target[0]) # label 9

# import matplotlib.pyplot as plt
# plt.imshow(train_input[0],cmap='gray')  # label 9인  input 이미지 체크
# plt.show()
# 훈련데이터 스케일 0~255 사이의 데이터를 0~1 사이로 변환
# 합성곱 층은 3차원 데이터 입력 기대
train_scaled = train_input.reshape(-1,28,28,1) / 255.0
print(train_scaled.shape)

from sklearn.model_selection import train_test_split
#  훈련 세트와 검증 세트로 분할
train_scaled, val_scaled, train_target , val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)
print(len(train_scaled)) # 48000
print(len(val_scaled)) # 12000

from tensorflow.keras.models import load_model
CNN_NewModel = load_model('best-cnn-model.h5')
#예측
preds = CNN_NewModel.predict(val_scaled)
print(preds[0]) # 검증 데이터로 예측한 첫번째 결과값
pred_label = np.argmax(preds[0])  # 예측 결과값에 대한 최대 예측값(약 1) 인덱스 추출
print(pred_label)
print(val_target[0]) # 검증 데이터의 타깃값 확인

# #모델 성능 평가
# val_loss , val_accuracy = CNN_NewModel.evaluate(val_scaled, val_target)
# print('\nval_loss : %.4f, val_accuracy : %.4f' %(val_loss, val_accuracy))

# 임의의 패션 이미지 데이터 load 후 모델 분류 예측
import cv2  # opencv-python 4.4.0.46 버전 설치
import numpy as np

img_path = 'shoes_3.jpg'
img = cv2.imread(img_path,0)  # flag 0 일 경우 Grayscale로 이미지 읽어들임
# cv2.imshow('sandal', img)
# # print(img)
# k = cv2.waitKey(0)  # 이미지 출력 상태로 임의의 키 입력할때 까지 무한 대기
# if k == 27:  # esc key 입력
#     cv2.destroyAllWindows()

#print(img.shape)
# (28, 28) 크기로 이미지 사이즈 조정
img_resize = cv2.resize(img, dsize=(28,28),interpolation=cv2.INTER_AREA)
# cv2.imshow('sandal_resize',img_resize)
# print(img_resize)
# cv2.waitKey(0)  # 이미지 출력 상태로 임의의 키 입력할때 까지 무한 대기
# cv2.destroyAllWindows()
print(img_resize.shape)
# 이미지 gray색상을 훈련 데이터셋과 일치시키기 위해
# gray색상을 반전 시켜줘야함
# (이미지 바탕색인 흰색(255)를 --> 검정색(0)으로 반전 )
img_reverted = cv2.bitwise_not(img_resize)
#print(img_reverted)
cv2.imshow('cnv_sandal', img_reverted)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()

new_img = img_reverted / 255.0 # 스케일 변환
print(new_img)
new_img = new_img.reshape(1,28,28,1) # 예측 입력을 위한 shape 변환

preds = CNN_NewModel.predict(new_img[0:1])

# numpy 데이터 소숫점 이하 3자리까지 출력 및 과학적 표기법 억제
np.set_printoptions(precision=3, suppress=True)
print(preds)

# # import matplotlib.pyplot as plt
# # plt.bar(range(0,10), preds[0])
# # plt.xlabel('class')  # 카테고리 클래스
# # plt.ylabel('prob.')  # 예측 추정치
# # plt.show()

classes = ['티셔츠', '바지', '스웨터', '드레스','코트', '샌달', '셔츠', '스니커즈', '가방', '앵클부츠']
# preds 가장 큰 인덱스 찾아 리스트 인덱스 색인
print(classes[np.argmax(preds)])

# #test 셋 데이터 평가
# test_scaled = test_input.reshape(-1, 28, 28, 1) / 255.0

# print('\n Tes Accuracy : %.4f' %(CNN_NewModel.evaluate(test_scaled, test_target,verbose=0)[1]) )
