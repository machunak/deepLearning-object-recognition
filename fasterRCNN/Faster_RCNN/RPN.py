from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input
import tensorflow as tf

class RPNCls():
    def create_model(self, k, numberofchannels):

        input = Input(shape=(None,None,numberofchannels))

        # kernel_initializer = 'normal' : 커널 필터 초기 설정 ==> 가우시안 분포
        # intermediate layer 설계
        #  name='레이어이름설정' ==> 이름 문자열에 공백 들어가면 안됨
        out = Conv2D(512, (3,3), padding='same', activation='relu', kernel_initializer='normal',
                     name='intermediate_layer')(input)
        # output shape :  [(None, None, None, 512)]
        print('out shape : ', out.shape)
        #kernel_initializer = 'uniform' ==> 균일분포
        #1*1 커널 conv, 2*k개의 필터 적용효과 ==> w, h 유지하면서 output 차원수가 2*k 필터로 됨(차원축소효과)
        #classification layer ( 개체유무 예측 정보 출력 )
        class_output = Conv2D(2*k, (1,1), activation='sigmoid', kernel_initializer='uniform',
                              name='classification_layer')(out)
        # class_output = Conv2D(2*k, (1,1), activation='softmax', kernel_initializer='uniform',
        #                       name='classification_layer')(out)
        # kernel_initializer='zero' : 가중치 초기화 방법 ==> 모든 가중치 0으로 초기화
        # regressor layer ( 앵커 위치 예측 정보 출력 )
        regressor_output = Conv2D(4*k,(1,1), activation='linear', kernel_initializer='zero',
                                  name='regressor_layer')(out)

        rpn_model = Model(inputs = input, outputs = [class_output, regressor_output])


        rpn_model.compile(optimizer='adam', loss=[tf.keras.losses.BinaryCrossentropy(), tf.compat.v1.losses.huber_loss])
        #rpn_model.compile(optimizer='adam', loss=['categorical_crossentropy', tf.compat.v1.losses.huber_loss])

        print('=============== rpn model complete ================')
        rpn_model.summary()
        return rpn_model



