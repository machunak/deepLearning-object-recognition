from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten
import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_tran, x_test = x_train / 255.0 , x_test / 255.0
print(len(x_tran), len(y_train))
print(y_train.shape)

# 차원 1 증가 ( axis = 0, axis = 1, axis = -1 )
x_tran = tf.expand_dims(x_tran, -1) # `axis=-1` adds an inner most dimension:
print(x_tran.shape) # (60000, 28, 28) ==> (60000, 28, 28, 1)
x_test = tf.expand_dims(x_test, -1) # `axis=-1` adds an inner most dimension:
print(x_test.shape) # (10000, 28, 28) ==> (10000, 28, 28, 1)



class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size = 32, shuffle=True):
        self.batch_size = batch_size
        self.x, self.y = x_set, y_set
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.x))

    def __len__(self):  # 시퀀스의 배치 수  ( 일괄처리의 개수 )
        #  step size(length of data / batch size) 리턴
        # 예)  100 dataset 길이 /  20 batch size  ==> 5 리턴 ( 5회 __getitem__ 호출하게됨 )

        # 즉, __getitem__() 메소드 총 호출 회수가 되며
        # fit()동작시 step size-1 이하의 값중의 하나의 index를 추출해 __getitem__()에서
        # 추출할 dataset의 시작 index사용함으로 주어진값보다 크 가장 가까운 정수로
        #  return할 step size를 표현해야함
        print('__len__')
        return int( np.ceil( len(self.x) /float(self.batch_size) ) )

    # def on_epoch_end(self):  # 1 epoch 끝 동작 마다 호출되는 함수
    #     # 한 epoch 이후 인덱스 서플
    #     print('on_epoch_end')
    #     if self.shuffle:
    #         np.random.shuffle(self.indexes)


    def __getitem__(self, index):  # index ==> 추출할 데이터셋 배치의 시작위치
        print('indx : ', index)
        batch_x = self.x[index * self.batch_size : (index+1)*self.batch_size]
        batch_y = self.y[index * self.batch_size : (index+1)*self.batch_size]
        return batch_x, batch_y

# DataGenerator 객체생성
generator = DataGenerator(x_tran, y_train, batch_size=32)

# Model layer 정의
input = Input(shape=(28,28,1))
x = Conv2D(32, 3, activation='relu')(input)
x = Conv2D(64, 3, activation='relu')(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
output = Dense(10, activation='softmax')(x)

# model 정의
model = tf.keras.models.Model(input, output)
model.summary()

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy())

# shuffle=True 일 경우 __getitem__(self, index)에 전달되는
# index는  __len__()의 반환값인 step size -1 값 중 랜덤 index를 전달
# shuffle=False 일 경우 1 epoches 당 0 부터  __len()__의 리턴값인 step size - 1 까지
# 순차적인 index를 __getitem__() 로 전달

model.fit(x=generator, epochs=30, shuffle=False)