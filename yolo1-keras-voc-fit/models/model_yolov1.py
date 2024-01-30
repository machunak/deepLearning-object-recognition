from tensorflow.keras.layers import Conv2D, MaxPooling2D, \
    Flatten, Dense, Reshape, LeakyReLU, Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate


def yolo_head(model_body, bbox_num=2, class_num=20):
    output = model_body

    xywhc_output = Conv2D(5 * bbox_num, 1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal')(output)
    p_output = Conv2D(class_num, 1,
                      padding='same',
                      activation='softmax',
                      kernel_initializer='he_normal')(output)

    outputs = concatenate([xywhc_output, p_output], axis=3)
    print('outputs shape : ', outputs.shape)
    return outputs

def Conv2D_BN_Leaky(input_tensor, *args):
    output_tensor = Conv2D(*args,
                           padding='same',
                           kernel_initializer='he_normal')(input_tensor)
    output_tensor = BatchNormalization()(output_tensor)
    output_tensor = LeakyReLU(alpha=0.1)(output_tensor)
    return output_tensor


def Backbone_darknet(input_tensor):
    conv1 = Conv2D_BN_Leaky(input_tensor, 64, 7, 2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D_BN_Leaky(pool1, 192, 3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D_BN_Leaky(pool2, 128, 1)
    conv3 = Conv2D_BN_Leaky(conv3, 256, 3)
    conv3 = Conv2D_BN_Leaky(conv3, 256, 1)
    conv3 = Conv2D_BN_Leaky(conv3, 512, 3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = pool3
    for _ in range(4):
        conv4 = Conv2D_BN_Leaky(conv4, 256, 1)
        conv4 = Conv2D_BN_Leaky(conv4, 512, 3)
    conv4 = Conv2D_BN_Leaky(conv4, 512, 1)
    conv4 = Conv2D_BN_Leaky(conv4, 1024, 3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D_BN_Leaky(pool4, 512, 1)
    conv5 = Conv2D_BN_Leaky(conv5, 1024, 3)
    conv5 = Conv2D_BN_Leaky(conv5, 512, 1)
    conv5 = Conv2D_BN_Leaky(conv5, 1024, 3)
    conv5 = Conv2D_BN_Leaky(conv5, 1024, 3)
    conv5 = Conv2D_BN_Leaky(conv5, 1024, 3, 2)

    conv6 = Conv2D_BN_Leaky(conv5, 1024, 3)
    conv6 = Conv2D_BN_Leaky(conv6, 1024, 3)

    return conv6


def model_yolov1(inputs):
    # 모델 설계시 strides=(2,2) 틀리면 메모리 할당 오류 뜸..

    darknet = Backbone_darknet(inputs)

    # 이부분에서 pretrained_darknet weights 있다면 darknet 에  set_weights()

    x = Flatten()(darknet)
    # 메모리 문제로 논문과 달리 512 dense사용
    x = Dense(512, activation='sigmoid')(x)
    x = Dense(1470, activation='sigmoid')(x)
    outputs = Reshape((7, 7, 30), dtype='float32')(x)

    # 논문과 달리 flatten(), reshape 사용하지 않고
    # (1,1) Conv 연산 활용 최종 output (7,7,30) 추출하는 방법도 가능
    # outputs = yolo_head(darknet)

    return outputs
