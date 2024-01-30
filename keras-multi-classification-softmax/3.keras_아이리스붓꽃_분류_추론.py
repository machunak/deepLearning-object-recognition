import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

pd.set_option('display.max_columns',100)
pd.set_option('display.width',1000)
iris = load_iris()
#print(iris)
print(iris['target_names'])
# sepal : 꽃받침,  petal : 꽃잎
# 'data' array : [6.4, 3.1, 5.5, 1.8] sepal_length, sepal_width, petal_length, petal_width
# 'target_names': array(['setosa', 'versicolor', 'virginica']
Iris_Data = pd.DataFrame(np.column_stack([iris['data'], iris['target']]),
                         columns=['sepal_len','sepal_wd',
                                  'petal_len','petal_wd','target'])
print(Iris_Data.info())

#훈련할 샘플 데이터 추출
X_data = Iris_Data[['sepal_len','sepal_wd','petal_len','petal_wd']]
print(X_data.head())
# 레이블(클래스) 추출
Y_target = Iris_Data['target']
print(Y_target.sample(10))  # target 값은  0, 1, 2 로 정수 인코딩 되어 있음.

##StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_data = scaler.fit_transform(X_data)

#정수인코딩된 타깃레이블 원-핫-인코딩 변환
from tensorflow.keras.utils import to_categorical
Y_onehot = to_categorical(Y_target.ravel())
print(Y_onehot)

from tensorflow.keras.models import load_model

bestmodel = load_model('bestmodel.h5')

# 모델 평가
print('accuracy : %.4f' %bestmodel.evaluate(X_data, Y_onehot)[1])

# 과학적 표기법 대신 소수점 6자리까지 나타냄
np.set_printoptions(precision=6, suppress=True)
pre = bestmodel.predict(X_data[0:1])
print('pre : ', pre)
print('target_onehot : ', Y_onehot[0])
print(iris['target_names'])

print(iris['target_names'][np.argmax(pre)])