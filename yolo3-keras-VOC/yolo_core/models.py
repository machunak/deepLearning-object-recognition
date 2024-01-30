import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, LeakyReLU, ZeroPadding2D, BatchNormalization, MaxPool2D
from tensorflow.keras.regularizers import l2
from configuration import *
from yolo_core.utils import read_class_names, bbox_giou, bbox_iou

STRIDES         = np.array(YOLO_STRIDES)
ANCHORS         = (np.array(YOLO_ANCHORS).T/STRIDES).T

class BatchNormalization(BatchNormalization):
    # "Frozen state" and "inference mode" are two separate concepts.
    # `layer.trainable = False` is to freeze the layer, so the layer will use
    # stored moving `var` and `mean` in the "inference mode", and both `gama`
    # and `beta` will not be updated !
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)

def DarknetConv2D(input_layer, filters_shape, downsample=False, activate=True, bn=True, activate_type='leaky'):
    if downsample:
        # ZeroPadding2D( ((top_pad,bottom_pad),(left_pad,right_pad)) )
        # top_pad 와 left_pad에 0의 행과 열을 하나 추가
        # (416, 416) --> (417, 417)
        input_layer = ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    conv = Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides,
                  padding=padding, use_bias=not bn, kernel_regularizer=l2(0.0005),
                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                  bias_initializer=tf.constant_initializer(0.))(input_layer)
    if bn:
        conv = BatchNormalization()(conv)
    if activate == True:
        if activate_type == "leaky":
            conv = LeakyReLU(alpha=0.1)(conv)
        elif activate_type == "mish":
            conv = mish(conv)
    return conv

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

def DarknetResidual(input_layer, input_channel, filter_num1, filter_num2, activate_type='leaky'):
    short_cut = input_layer
    conv = DarknetConv2D(input_layer, filters_shape=(1, 1, input_channel, filter_num1), activate_type=activate_type)
    conv = DarknetConv2D(conv       , filters_shape=(3, 3, filter_num1,   filter_num2), activate_type=activate_type)

    residual_output = short_cut + conv
    return residual_output

def upsample(input_layer):
    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='nearest')

def darknet53(input_data):
    input_data = DarknetConv2D(input_data, (3, 3,  3,  32))
    input_data = DarknetConv2D(input_data, (3, 3, 32,  64), downsample=True)

    for i in range(1):
        input_data = DarknetResidual(input_data,  64,  32, 64)

    input_data = DarknetConv2D(input_data, (3, 3,  64, 128), downsample=True)

    for i in range(2):
        input_data = DarknetResidual(input_data, 128,  64, 128)

    input_data = DarknetConv2D(input_data, (3, 3, 128, 256), downsample=True)

    for i in range(8):
        input_data = DarknetResidual(input_data, 256, 128, 256)

    route_1 = input_data
    input_data = DarknetConv2D(input_data, (3, 3, 256, 512), downsample=True)

    for i in range(8):
        input_data = DarknetResidual(input_data, 512, 256, 512)

    route_2 = input_data
    input_data = DarknetConv2D(input_data, (3, 3, 512, 1024), downsample=True)

    for i in range(4):
        input_data = DarknetResidual(input_data, 1024, 512, 1024)

    return route_1, route_2, input_data

def YOLOv3_model(input_layer, NUM_CLASS):
    #print("YOLOv3_model")
    # Darknet53 network 활용해 3개의 branch layer 얻음
    route_1, route_2, conv = darknet53(input_layer)

    conv = DarknetConv2D(conv, (1, 1, 1024, 512))  # conv2d_52
    conv = DarknetConv2D(conv, (3, 3, 512, 1024))  # conv2d_53
    conv = DarknetConv2D(conv, (1, 1, 1024, 512))  # conv2d_54
    conv = DarknetConv2D(conv, (3, 3, 512, 1024))  # conv2d_55
    conv = DarknetConv2D(conv, (1, 1, 1024, 512))  # conv2d_56
    conv_lobj_branch = DarknetConv2D(conv, (3, 3, 512, 1024)) # conv2d_57

    # conv_lbbox is used to predict large-sized objects , Shape = [None, 13, 13, 255]
    # large 크기 객체 예측용으로 사용 Shape = [None, 13, 13, 255] OR [None, 13, 13, 75]
    # conv2d_58 ( 전이  skip )
    conv_lbbox = DarknetConv2D(conv_lobj_branch, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = DarknetConv2D(conv, (1, 1, 512, 256))  # conv2d_59
    # upsample here uses the nearest neighbor interpolation method, which has the advantage that the
    # upsampling process does not need to learn, thereby reducing the network parameter
    conv = upsample(conv)

    conv = tf.concat([conv, route_2], axis=-1)  #  (None, 26, 26, 768)

    conv = DarknetConv2D(conv, (1, 1, 768, 256))  # conv2d_60
    conv = DarknetConv2D(conv, (3, 3, 256, 512))  # conv2d_61
    conv = DarknetConv2D(conv, (1, 1, 512, 256))  # conv2d_62
    conv = DarknetConv2D(conv, (3, 3, 256, 512))  # conv2d_63
    conv = DarknetConv2D(conv, (1, 1, 512, 256))  # conv2d_64
    conv_mobj_branch = DarknetConv2D(conv, (3, 3, 256, 512)) # conv2d_65

    # conv_mbbox is used to predict medium-sized objects, shape = [None, 26, 26, 255]
    # medium 크기 객체 예측용으로 사용 Shape = [None, 26, 26, 255] OR [None, 26, 26, 75]
    # conv2d_66 ( 전이 skip )
    conv_mbbox = DarknetConv2D(conv_mobj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = DarknetConv2D(conv, (1, 1, 256, 128)) # conv2d_67
    conv = upsample(conv)

    conv = tf.concat([conv, route_1], axis=-1)  # (None, 52, 52, 384)

    conv = DarknetConv2D(conv, (1, 1, 384, 128))  # conv2d_68
    conv = DarknetConv2D(conv, (3, 3, 128, 256))  # conv2d_69
    conv = DarknetConv2D(conv, (1, 1, 256, 128))  # conv2d_70
    conv = DarknetConv2D(conv, (3, 3, 128, 256))  # conv2d_71
    conv = DarknetConv2D(conv, (1, 1, 256, 128))  # conv2d_72
    conv_sobj_branch = DarknetConv2D(conv, (3, 3, 128, 256)) # conv2d_73

    # conv_sbbox is used to predict small size objects, shape = [None, 52, 52, 255]
    # small 크기 객체 예측용으로 사용 Shape = [None, 52, 52, 255] OR [None, 52, 52, 75]
    # conv2d_74 ( 전이 skip )
    conv_sbbox = DarknetConv2D(conv_sobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]

def Create_Yolov3(input_size=416, channels=3, training=False, CLASSES=YOLO_COCO_CLASSES):
    NUM_CLASS = len(read_class_names(CLASSES))
    #print("NUM_CLASS : ", NUM_CLASS)

    input_layer = Input([input_size, input_size, channels]) # 416x416x3
    conv_tensors = YOLOv3_model(input_layer, NUM_CLASS)

    # training 과정은 output으로 마지막의 conv_tensors + decoding된 pred_tensors 를 동시 반환
    output_tensors = []
    # for i, conv_tensor in enumerate(conv_tensors):
    #     if training:
    #         output_tensors.append(conv_tensor)
    #         pred_tensor = decode(conv_tensor, NUM_CLASS, i)
    #         output_tensors.append(pred_tensor)
    #     else:
    #         output_tensors.append(conv_tensor)
    for i, conv_tensor in enumerate(conv_tensors):
        pred_tensor = decode(conv_tensor, NUM_CLASS, i)
        if training: # 학습 과정일때는 conv_tensor 와 pred_tensor 필요로 함
            output_tensors.append(conv_tensor)
        output_tensors.append(pred_tensor)

    #print('output_tensors : ')
    #print(output_tensors)
    yolov3_model = tf.keras.Model(input_layer, output_tensors)
    return yolov3_model


def decode(conv_output, NUM_CLASS, i=0):
    #print('decoding!!')
    # shape=(None, 52, 52, 75) ==> shape=(None, None, None, 3, 25) 변환
    # shape=(None, 26, 26, 75) ==> shape=(None, None, None, 3, 25) 변환
    # shape=(None, 13, 13, 75) ==> shape=(None, None, None, 3, 25) 변환
    conv_shape = tf.shape(conv_output)
    # print(' conv_shape : ')
    # print(conv_shape)
    batch_size       = conv_shape[0]
    output_size      = conv_shape[1]
    #print('output_size : ', output_size)
    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    # 모델 출력값 분리 추출
    conv_raw_txty = conv_output[:,:,:,:,0:2] # 모델 출력 box 중심 좌표
    conv_raw_twth = conv_output[:,:,:,:,2:4] # 모델 출력  box 너비,높이
    conv_raw_conf = conv_output[:,:,:,:,4:5] # 모델 출력의 객체 존재 확신(신뢰)도
    conv_raw_prob = conv_output[:,:,:,:,5: ] # 모델 출력의 class분류 추청도

    # grid 좌표 데이터를 생성하기위해 output_size별 grid를 생성(np.meshgrid 와 유사)
    y=tf.range(output_size, dtype=tf.int32)
    # print('y : ',y) ==> tf.range(52, dtype=tf.int32) 일 경우 ==> [0 1 2 .... 51]
    y = tf.expand_dims(y, -1)
    y = tf.tile(y, [1, output_size]) # (None,1) ==> [1,None] 만큼 곱에서 shape변경(열축 확장)
    # ==> (output_size, output_size)

    x=tf.range(output_size, dtype=tf.int32)
    x = tf.expand_dims(x, 0)  # axis = 0 ==>첫번째 축에 차원 추가
    #print('x : ', x)
    x = tf.tile(x, [output_size, 1]) # (1,None) ==> [None,1] 만큼 곱에서 shape변경(행축 확장)
    #print('x : ', x)
    # ==> (output_size, output_size)
    #print(x[:,:,tf.newaxis]) # output_size가 52일 경우 ==> [52행 1열]을 52개로 묶어 표현 (52, 52, 1)
    #print(y[:, :, tf.newaxis])
    xy_grid = tf.concat([ x[:,:,tf.newaxis],y[:,:,tf.newaxis]], axis=-1)
    #print('xy_grid_con_shape : ', xy_grid.shape)
    # xy_grid를 batch_size 와 anchor수(3)으로 확장
    # 전체 그리드 좌표점 마다 3개의 anchor box를 생성함으로  xy_grid 좌표점
    # anchor 수 만큼 확장
    xy_grid = tf.tile(xy_grid[tf.newaxis,:,:,tf.newaxis,:],[batch_size,1,1,3,1])
    # float32 로 타입변환
    xy_grid = tf.cast(xy_grid, tf.float32)



    # 모델 출력값에 sigmoid 적용후 각 gridcell 좌표 더해 예측 중심좌표를 구하고
    # 예측 중심 좌표에 각 stride를 곱해 원래 크기 좌표로 복원 ( label이 원 크기 좌표 임으로 )
    # conv_raw_txty ==> (batch_size, output_size, output_size, 3 anchor, txty )
    # xy_grid       ==> (batch_size, output_size, output_size, 3,          2 )
    pred_xy = (tf.sigmoid(conv_raw_txty) + xy_grid) * STRIDES[i]

    # 예측 boxes 의 너비,높이 계산
    # 예측 boxes 의 너비 높이 가 학습에 의해 GT 에 가깝도록 conv_raw_twth 가중치 갱신(학습)
    # 임의의 bounding box의 너비와높이를 생성하도록 학습하는것 보다
    # 정규화된 prior anchor box를 활용함으로서 학습이 안정적으로 되게 함
    pred_wh = (tf.exp(conv_raw_twth) * ANCHORS[i]) * STRIDES[i]
    # xywh로 병합
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    # object box의 예측 신뢰도 sigmoid 적용
    pred_conf = tf.sigmoid(conv_raw_conf)
    # 예측 클래스 추정치 sigmoid 적용
    pred_prob = tf.sigmoid(conv_raw_prob)

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

def compute_loss(pred, conv, label, bboxes, i=0, CLASSES = YOLO_COCO_CLASSES):
    NUM_CLASS = len(read_class_names(CLASSES))
    #print("NUM_CLASS : ", NUM_CLASS)
    conv_shape = tf.shape(conv)
    #print('conv_shape : ', conv_shape)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    #input_size = STRIDES[i] * output_size
    #print('input size : ', input_size) # 416
    #print('label shape : ', label.shape)
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3 , 5+NUM_CLASS))

    conv_raw_conf = conv[:,:,:,:,4:5] # 모델 출력 Confidence score
    conv_raw_prob = conv[:,:,:,:,5:]  # 모델 출력 클래스분류 추정도

    pred_xywh = pred[:,:,:,:,0:4]  # 원래 크기 좌표로 Decoding 한 예측 xywh
    pred_conf = pred[:, :, :, :, 4:5]  # Decoding한 예측  Confidence score

    label_xywh   = label[:,:,:,:,0:4] # 타깃 wywh
    # 각 stride 그리드 셀별 3개의 anchor 기준으로 labeling 된 bounding box 좌표
    #print('label_xywh shape : ', label_xywh.shape)  # (1, 26, 26, 3, 4)
    respond_bbox = label[:,:,:,:,4:5] # 타깃 Confidence score
    label_prob   = label[:,:,:,:,5:] # 클래스분류 smooth_onehot 라벨

    # pred_xy = pred_xywh[:,:,:,:,:2]
    # label_xy = label_xywh[:,:,:,:,:2]
    # pred_wh = pred_xywh[:, :, :, :, 2:4]
    # label_wh = label_xywh[:, :, :, :,2:4]
    # boxesxy_loss = respond_bbox * tf.square(pred_xy - label_xy)
    # boxeswh_loss = respond_bbox * tf.square(pred_wh - label_wh)
    # boxes_loss = 5.0 * ( boxesxy_loss + boxeswh_loss)
    # 1차원
    giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
    #print('giou.shape : ', giou.shape)  # (1, 52, 52, 3,1) -> (1, 26, 26, 3,1) -> (1, 13, 13, 3,1)
    #input_size = tf.cast(input_size, tf.float32) # 416을 float형태로

    # bbox_loss_scale ==> 안 해줘도 될듯한데..
    # 이미지속 객체가 작을수록 못 찾는 확률이 클수 있음으로 loss를 좀더 많이 반영토록함
    #bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    #print('bbox_loss_scale : ', bbox_loss_scale) # 이미지 사이즈 속 객체 크기에 대한 loss 보정

    # loss에 책임 있는 box의 giou loss 계산
    giou_loss = respond_bbox  * (1 - giou)
   # giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

    # (1, 52, 52, 3, 1) --> (1, 26, 26, 3, 1) --> ( 1, 13, 13, 3, 1)
    #print('giou_loss.shape :', giou_loss.shape)
    # pred_xywh[:,:,:,:,np.newaxis,:] ==> (1, 26, 26, 3, 4) --> (1, 26, 26, 3, 1, 4)
    # bboxes.shape (1, 100, 4) (batch_size, box수, box좌표 ) ==> (1, 1, 1, 1, 100, 4)

    # 배경 loss를 계산하기 위해 IOU 연산량을 줄이기 위해 GT boxes 좌표를 별도로 모아
    # 예측 좌표와 IOU 계산해서 배경 Loss 밸런싱 파라미터 구함
    #print('bboxes shape : ', bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :].shape)
    #print('pred_xywh.shape : ', pred_xywh[:,:,:,:,np.newaxis,:].shape)

    # label 객체확신도가 0 이면서
    # 예측과 label box 간 iou 계산하여 각 그리드 지점의
    # 3개의 예측 box의 최대 iou가 YOLO_IOU_LOSS_THRESH < 0.5 경우
    # 배경 loss에 책임있는 box로 선정
    iou = bbox_iou(pred_xywh[:,:,:,:,np.newaxis,:], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    #print('iou shape : ', iou.shape) #  (1, 26, 26, 3, 100)
    #print(tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1).shape) # (1, 26, 26, 3, 1)
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)
    # max_iou 가 YOLO_IOU_LOSS_THRESH 보다 작다면 객체를 포함하지 않는 배경 box로 취급
    # max_iou < YOLO_IOU_LOSS_THRESH ==> True(배경박스), True ==> 1.0 으로 변경
    #print(tf.cast(max_iou < YOLO_IOU_LOSS_THRESH, tf.float32))
    # 배경 boxes loss의 밸런싱 파라미터값으로 사용 ==> respond_bgd

    # 객체확신도가 0 이면서 iou < 0.5 인 box만 배경 box로 취급해 loss function 수행
    respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < YOLO_IOU_LOSS_THRESH, tf.float32)
    #print('respond_bgd shape : ', respond_bgd.shape)


    # 배경, 전경 confidence loss ==> focal loss 적용
    # 배경, 전경 class 불균형에 대한 loss 문제를 해결하기위해
    # 낮은 loss를 갖는 배경(easy sample)의 loss 감소폭을 더 높여 easy sample은
    # 빠르게 학습 완료되게하고  높은 loss를 갖는 전경(Hard sample)
    # 에 초점(focal)을 두어 전경 학습에 더 집중하도록 하는 원리
    # focal loss 참조 사이트  ==> https://woochan-autobiography.tistory.com/929
    #                        ==> https://gaussian37.github.io/dl-concept-focal_loss/

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)  # focal loss 공식, 2 ==> r

    conf_loss = conf_focal * (
        respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        +
        respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )

    #conf_focal = tf.pow(respond_bbox - pred_conf, 2)  # focal loss 공식, 2 ==> r

    # conf_loss =  (
    #     respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    #     +
    #     respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    # )

    # class 분류 ==> sigmod + cross entroy loss 적용
    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    # tf.reduce_sum(giou_loss, axis=[1,2,3,4])
    # ==> batch단위 축을 제외한 모든 요소를 합하면서 차원 축소
    # tf.reduce_mean() ==> batch단위 요소의 평균 계산
    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

    return giou_loss, conf_loss, prob_loss




