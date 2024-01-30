# 사용 libraray import
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

import numpy as np
import tensorflow as tf
from yolo_core.dataset import Dataset
from yolo_core.models import Create_Yolov3, compute_loss
import sys
from configuration import *
from yolo_core.utils import load_yolo_darknetweights

# 전이학습 시킬 darknet_weights 가중치 파일 경로 설정
Darknet_weights_file = YOLO_V3_WEIGHTS  # ./checkpoints/yolov3.weights

def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f'GPUs {gpus}')
    if len(gpus) > 0:
        try:
            #print('gpu growth')
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError:
            pass

    # train / test Dataset 설정
    trainset = Dataset('train')
    steps_per_epoch = len(trainset)
    #print('steps_per_epoch : ', steps_per_epoch)  # 148
    # 학습 진행에 따른 학습률 적용하기 위한 학습 진행 회수 체크 변수
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = TRAIN_WARMUP_EPOCHS * steps_per_epoch
    total_steps = TRAIN_EPOCHS * steps_per_epoch

    # 전이학습을 위한 coco data 학습된 yolov3 darknet 가중치 로딩
    if TRAINDATA_trainsfer:
        # coco dataset 으로 학습된 darknet 가중치 로딩
        # 밑에서 최종 모델에 여기서 로딩된 dartknet 가중치를 읽어와 전이 해줌(전이학습)
        Darknet = Create_Yolov3(input_size=YOLO_INPUT_SIZE, CLASSES=YOLO_COCO_CLASSES)
        #Darknet.summary()
        # darknet 가중치 로딩

        load_yolo_darknetweights(Darknet,Darknet_weights_file)

    # voc2012.names train data set으로  학습 모델 재 설계
    yolov3_model = Create_Yolov3(input_size=YOLO_INPUT_SIZE,training=True, CLASSES=TRAIN_CLASSES)
    #yolov3_model.summary()

    # 로딩된 Darknet 가중치 설정
    for i, l in enumerate(Darknet.layers):
        layer_weights = l.get_weights()
        if layer_weights != []:
            try:
                yolov3_model.layers[i].set_weights(layer_weights)
                #print('set_weigths ', yolov3_model.layers[i].name)
                #print('trainable ',  yolov3_model.layers[i].trainable)
            except:
                print('skipping', yolov3_model.layers[i].name)

    yolov3_model.summary()  # final model
    save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, DATA_TYPE)
    print('save dir : ', save_directory)
    optimizer = tf.keras.optimizers.Adam()
    yolov3_model.compile(optimizer=optimizer) # 모델 환경 설정 끝

    def train_step(image_data, target):
        with tf.GradientTape() as tape:
            #print('GradiantTape')
            pred_result = yolov3_model(image_data, training=True) # 모델 예측
            # [conv_tensor,pred_tensor,conv_tensor,pred_tensor,conv_tensor,pred_tensor] 반환
            #print(pred_result)
            #print('pred_result Len : ', len(pred_result))
            giou_loss = conf_loss = prob_loss = 0
            grid = 3
            for i in range(grid):
                conv, pred = pred_result[i*2],pred_result[i*2+1]
                # target ==> (batch_smaller_target, batch_medium_target, batch_larger_target)
                # batch_smaller_target ==> batch_label_sbbox, batch_sbboxes
                loss_items = compute_loss(pred, conv, *target[i], i, CLASSES = TRAIN_CLASSES)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            gradients = tape.gradient(total_loss, yolov3_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, yolov3_model.trainable_variables))

            # 안정적인 학습 진행을 위해
            # update learning rate
            # https://norman3.github.io/papers/docs/bag_of_tricks_for_image_classification.html 참조
            # about warmup: https://arxiv.org/pdf/1812.01187.pdf&usg=ALkJrhglKOPDjNt6SHGbphTHyMcT0cuMJg
            global_steps.assign_add(1)  # 인자로 주어진 만큼 더함
            if global_steps < warmup_steps:  # and not TRAIN_TRANSFER:
                lr = global_steps / warmup_steps * TRAIN_LR_INIT
            else:
                lr = TRAIN_LR_END + 0.5 * (TRAIN_LR_INIT - TRAIN_LR_END) * (
                    (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi)))
            optimizer.lr.assign(lr.numpy())

        return global_steps.numpy(), optimizer.lr.numpy(), giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()


    
    for epoch in range(TRAIN_EPOCHS): # 몇 번 반복학습 시킬건지
        count, giou_val, conf_val, prob_val, total_val = 0, 0, 0, 0, 0

        for image_data, target in trainset:  # for() 한번 동작 시 마다 --> __next__(self) 동작
            results = train_step(image_data, target)
            count += 1
            print('train_steping count : ', count)
            giou_val += results[2]
            conf_val += results[3]
            prob_val += results[4]
            total_val += results[5]

            print('======== train one batch step finish ========')

        print('=============train one Epoch finish ==========')
        print("Train Loss per Epoch  // giou:{:7.2f}, conf:{:7.2f}, prob:{:7.2f}, total:{:7.2f}".
              format(giou_val / count, conf_val / count, prob_val / count, total_val / count))

        # voc data 재학습에 대한 학습 모델 저장 ( 저장 경로 불일치시 저장 오류 )
        #yolov3_model.save_weights(save_directory + '\myyolov3.weights.h5')
        yolov3_model.save(save_directory + '\yolov3_first.h5')


if __name__ == '__main__':
    main()
