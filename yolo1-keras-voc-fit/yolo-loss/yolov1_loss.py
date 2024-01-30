import tensorflow.keras.backend as K
from tensorflow.keras.layers import concatenate
import tensorflow as tf


def xywh2minmax(xy, wh):
    xy_min = xy - (wh / 2)  # 중심좌표를==>xmin,ymin 로 변환
    xy_max = xy + (wh / 2)  # 중심좌표를==>xmax,ymax 로 변환
    #tf.print('xy_min , xy_max ', xy_min[0][3][3], xy_max[0][3][3]) # xy_min , xy_max  [[[139 130]]] [[[314 363]]]
    return xy_min, xy_max


def Iou_Cal(pred_mins, pred_maxes, true_mins, true_maxes):
    intersect_mins = K.maximum(pred_mins, true_mins)
    intersect_maxes = K.minimum(pred_maxes, true_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    pred_wh = pred_maxes - pred_mins
    true_wh = true_maxes - true_mins
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas

    return iou_scores


def yolo_dataset_decoding(feats):
    # Dynamic implementation of conv dims for fully convolutional model.
    # label_box_xywh shape ==> [None, 7, 7, 1, 4]
    conv_dims = K.shape(feats)[1:3]  # assuming channels last
    #tf.print(conv_dims) # [ 7 7 ]
    # In YOLO the height index is the inner most iteration.
    conv_height_index = K.arange(0, stop=conv_dims[0])
    conv_width_index = K.arange(0, stop=conv_dims[1])
    conv_height_index = K.tile(conv_height_index, [conv_dims[1]])
    #tf.print(conv_height_index) #[0 1 2 3 4 5 6 ] * 7 곱해서줌 ==> [0 1 2 3.....6]

    # conv_width_index = K.repeat_elements(conv_width_index, conv_dims[1], axis=0)
    #  K.expand_dims(conv_width_index, 0) ==> [0 1 2 3 4 5 6 ] 를 행단위 7 곱하기 위해 해줌
    conv_width_index = K.expand_dims(conv_width_index, 0)
    conv_width_index = K.tile( conv_width_index, [conv_dims[0], 1])
    #tf.print(conv_width_index)  # ==> [0 1 2 ...6] * [7 , 1] ==> (7행, 7열)
    #tf.print(K.transpose(conv_width_index))
    conv_width_index = K.transpose(conv_width_index)
    conv_width_index = K.flatten(conv_width_index)
    #tf.print(conv_width_index.shape)
    #tf.print(conv_width_index)
    conv_index = K.stack([conv_height_index, conv_width_index])
    conv_index = K.transpose(conv_index)
    #tf.print(conv_index)
    conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
    #tf.print(conv_index)
    conv_index = K.cast(conv_index, K.dtype(feats))
    #tf.print(conv_index)
    conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))
    #tf.print(conv_dims) # ==> [[[[[7 7]]]]]
    # label_box_xywh 의 xy 데이터에 각 grid cell 좌표점(conv_index)을 더해줌
    #tf.print(feats[..., :2].shape)
    # xy 좌표를 복원하기 위해 / [ 7 7 ] 해주고 448 곱해줌
    box_xy = (feats[..., :2] + conv_index) / conv_dims * 448
    box_wh = feats[..., 2:4] * 448

    return box_xy, box_wh


def yolov1_loss_func(y_true, y_pred):
    # label 데이터 추출 분리
    label_class = y_true[..., 5:]  # None * 7 * 7 * 20
    label_box_xywh = y_true[..., :4]  # None * 7 * 7 * 4
    # real label :  [array([139, 130, 314, 363,   6])] <--  첫 batch_dataset의  첫 라벨 xmin,ymin,xmax,ymax,class
    # [139, 130, 314, 363] ==> Encoding과 동시 [3][3]에 저장되어 있음
    #tf.print(label_box_xywh[0][3][3]) #==> 첫 batch_dataset의  첫 라벨 x,y,w,h Encoding 좌표 데이터
    #[0.5390625 0.8515625 0.390625 0.520089269] <-- x,y,w,h Encoding 좌표

    # 객체 존재 신뢰도(confidence) 추출
    response_mask = y_true[..., 4]  # None * 7 * 7
    response_mask = K.expand_dims(response_mask)  # None * 7 * 7 * 1
    #tf.print('response_mask shape : ', response_mask.shape)


    # 7x7x25 label과 달리 7x7x30 의 예측 데이터 추출 분리
    predict_class = y_pred[..., 10:]  # None * 7 * 7 * 20
    predict_trust1 = y_pred[..., 4]  # None * 7 * 7 * 1
    predict_trust2 = y_pred[..., 9]  # None * 7 * 7 * 1
    # 이후 추론단계에서는 예측 box 2개 별
    # predict_trust1 * predict_class 과  predict_trust2 * predict_class
    # 로 NMS 수행
    predict_trust =  K.stack([predict_trust1, predict_trust2],axis=3) # None*7*7*2
    #tf.print(' predict_trust : ' , predict_trust[0][3][3])

    predict_box1 = y_pred[..., :4]  # None * 7 * 7 * 4
    predict_box2 = y_pred[..., 5:9]  # None * 7 * 7 * 4
    predict_box = K.stack([predict_box1, predict_box2],axis=3)
    #tf.print(predict_box.shape)  #  [None, 7, 7, 2, 4 ]

    # 좌표점 복원해서 minxy, maxxy 계산하기 위해 label, predict box shape 변경
    # label ==> None * 7 * 7 * 4      ==> [-1, 7, 7, 1, 4]
    # predict ==> [None, 7, 7, 2, 4 ] ==> [-1, 7, 7, 2, 4]
    _label_box = K.reshape(label_box_xywh, [-1, 7, 7, 1, 4])
    _predict_box = K.reshape(predict_box, [-1, 7, 7, 2, 4])

    # label 중심좌표 및너비,높이 좌표 복원
    label_xy, label_wh = yolo_dataset_decoding(_label_box)  # None * 7 * 7 * 1 * 2, None * 7 * 7 * 1 * 2
    #tf.print('label_xy, label_wh : ', label_xy[0][3][3], label_wh[0][3][3])
    label_xy = K.expand_dims(label_xy, 3)    # ? * 7 * 7 * 1 * 1 * 2
    label_wh = K.expand_dims(label_wh, 3)    # ? * 7 * 7 * 1 * 1 * 2
    # label_xy_exp = K.expand_dims(label_xy, 3)  # None * 7 * 7 * 1 * 1 * 2
    # label_wh_exp = K.expand_dims(label_wh, 3)  # None * 7 * 7 * 1 * 1 * 2
    #tf.print(label_xy.shape, label_wh.shape)
    # 복원된 label 중심및너비,높이 좌표를  label의 min xy, max xy 좌표로 변환
    label_xy_min, label_xy_max = xywh2minmax(label_xy, label_wh)  # None * 7 * 7 * 1 * 1 * 2, None * 7 * 7 * 1 * 1 * 2

    #tf.print(label_xy_min[0][3][3], label_xy_max[0][3][3])
    #[[[139 130]]] [[[314 363]]] <-- 복원된 첫 라벨 데이터 좌표

    # 두 예측 좌표를 복원해서 ==> 두개의 min xy, max xy 좌표로 변환
    predict_xy, predict_wh = yolo_dataset_decoding(_predict_box)  # None * 7 * 7 * 2 * 2, None * 7 * 7 * 2 * 2
    predict_xy = K.expand_dims(predict_xy, 4)  # ? * 7 * 7 * 2 * 1 * 2
    predict_wh = K.expand_dims(predict_wh, 4)  # ? * 7 * 7 * 2 * 1 * 2
    # predict_xy_exp = K.expand_dims(predict_xy, 4)  # None * 7 * 7 * 2 * 1 * 2
    # predict_wh_exp = K.expand_dims(predict_wh, 4)  # None * 7 * 7 * 2 * 1 * 2
    predict_xy_min, predict_xy_max = xywh2minmax(predict_xy, predict_wh)  # None * 7 * 7 * 2 * 1 * 2, None * 7 * 7 * 2 * 1 * 2
    #tf.print(predict_xy_min[0][3][3], predict_xy_max[0][3][3])

    iou_scores = Iou_Cal(predict_xy_min, predict_xy_max, label_xy_min, label_xy_max)  # None * 7 * 7 * 2 * 1
    #tf.print('iou_scores shape  : ', iou_scores.shape)
    #tf.print(iou_scores[0][3][3])
    # best_ious ==> 두 box의 iou 값 [[44]
    #                               [40]]] 중 axis=4로 각각 최대값 찾아
    #                               ==> [44 40] 으로 차원 변환
    #
    #best_ious = K.max(iou_scores, axis=4)  # None * 7 * 7 * 2
    best_ious = K.squeeze(iou_scores, axis=4) # None * 7 * 7 * 2 로 차원 축소
    #tf.print('best_ious : ',best_ious.shape)
    #tf.print(best_ious[0][3][3]) # [0.00237091631 0.0212982688]

    # 차원 축소한 [44 40]  값중 큰값 선택 [44]
    max_iou = K.max(best_ious, axis=3, keepdims=True)  # None * 7 * 7 * 1
    #tf.print('max_iou : ', max_iou[0][3][3])
    # tf.cast() ==> Boolean형태인 경우 True이면 1, False이면 0을 출력
    # box_mask ==> [ 0 1 ] 로 변환  ==> loss에 책임있는  predictor 선택하기 위함
    box_mask = K.cast(best_ious >= max_iou, K.dtype(best_ious))  # None * 7 * 7 * 2
    # print('box_mask shape :', box_mask.shape)
    #tf.print('box_mask')
    #tf.print(box_mask[0][3][3])  # [0 1] tensor value값 확인

    # object_loss = box_mask * response_mask * K.square(1 - predict_trust)
    # predict_trust 예측값이 best_ious 값을 갖도록 loss 설정
    # 1 - predict_trust 설정은 객체가 존재하지만 초기 loss가 너무 커짐
    #  box_mask * response_mask ==> box_mask 가 1이고 response_mask 1인 위치의 loss만 계산
    object_loss = box_mask * response_mask * K.square(best_ious - predict_trust)
    no_object_loss = 0.5 * (1 - box_mask * response_mask) * K.square(0 - predict_trust)

    confidence_loss = object_loss + no_object_loss
    #tf.print('confidence_loss : ', confidence_loss.shape)
    confidence_loss = K.sum(confidence_loss)

     # response_mask ==> 객체가 존재하는 i번쨰 grid cell
    class_loss = response_mask * K.square(label_class - predict_class)
    class_loss = K.sum(class_loss)


    # 위에서 shape 변경한  _label_box 와 _predict_box 활용해 Encoding x,y,w,h 추출
    _label_box_xy = _label_box[...,:2]
    _label_box_wh = _label_box[...,2:4]
    # tf.print(_label_box_xy[0][3][3])
    # tf.print(_label_box_wh[0][3][3])
    _predict_box_xy = _predict_box[...,:2]
    _predict_box_wh = _predict_box[...,2:4]

    # 구지 yolo_dataset_decoding() 수행하지 않고 Encoding 된 xy로 loss 수행하면 안됨??
    #label_xy, label_wh = yolo_dataset_decoding(_label_box)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
    #predict_xy, predict_wh = yolo_dataset_decoding(_predict_box)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2

    # xywh loss 연산을 위해 box_mask, response_mask shape 변경하여 일치 시켜줌
    box_mask = K.expand_dims(box_mask)
    # tf.print(box_mask.shape)  # TensorShape([None, 7, 7, 2, 1])
    response_mask = K.expand_dims(response_mask)
    #tf.print(response_mask.shape)  # TensorShape([None, 7, 7, 1, 1])

    # yolo_dataset_decoding() 에서 448 크기로 boxes정보 복원해였음으로 다시 448로 나눠줌
    #tf.print(K.square((_label_box_xy - _predict_box_xy)).shape) #==> TensorShape([None, 7, 7, 2, 2])
    #tf.print('label_xy[3][3] : ', label_xy[0][3][3])
    #result = K.square((_label_box_xy - _predict_box_xy))
    #tf.print(result[0][3][3])
    box_loss = 5 * box_mask * response_mask * K.square( _label_box_xy - _predict_box_xy )
    box_loss += 5 * box_mask * response_mask * K.square( K.sqrt(_label_box_wh) - K.sqrt(_predict_box_wh) )
    #tf.print('label_xy : ',label_xy[0][3][3])
    #tf.print('predict_xy : ', predict_xy[0][3][3])
    #  Encoding xywh 데이터를 바로 SSE loss function 수행하는것보다
    #  복원한 xy 데이터를 448 원본이미지 크기로 정규화하여 SSE loss function 수행했더니 약간의 성능 개선
    # 추가 성능 개선을 위해 옵티마이저에 학습률 적용 필요
    # box_loss = 5 * box_mask * response_mask * K.square( (label_xy - predict_xy) / 448 )
    # box_loss += 5 * box_mask * response_mask * K.square( (K.sqrt(label_wh / 448) - K.sqrt(predict_wh / 448))  )

    box_loss = K.sum(box_loss)

    loss = confidence_loss + class_loss + box_loss

    return loss

