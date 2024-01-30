from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
from models.model_yolov1 import model_yolov1  # yolov1 모델설계
from data_sequence import SequenceData  # dataset Sequence loading 클래스
from yolo_loss.yolov1_loss import yolov1_loss_func  # loss function 설계
from tensorflow.keras.optimizers import  Adam  # 모델 최적화 알고리즘
import tensorflow as tf


def yolov1_main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f'GPUs {gpus}')
    if len(gpus) > 0:
        try:
            # 런타임 할당에 필요한 만큼의 GPU 메모리만 할당하려고 시도하며 메모리 증가 설정
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError:
            pass

    # batch_size 늘리거나 모델 설계 실수 시 GPU 메모리 부족 오류 발생
    epochs = 30
    batch_size = 4

    input_shape = (448, 448, 3)
    inputs = Input(input_shape)
    yolo_outputs = model_yolov1(inputs)

    model = Model(inputs=inputs, outputs=yolo_outputs)
    model.summary()

    # loss : nan 일 경우 여러 경우가 있으나 첫번쨰로 learning_rate 조절 필요
    optimizer = Adam(learning_rate=1e-4)

    model.compile(loss=yolov1_loss_func, optimizer=optimizer)

    save_dir = 'checkpoints'
    weights_path = os.path.join(save_dir, 'yolov1_weights.hdf5')
    checkpoint = ModelCheckpoint(weights_path, monitor='val_loss',
                                 save_weights_only=True, save_best_only=True)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=3, verbose=1)

    # if os.path.exists('checkpoints/yolov1_weights.hdf5'):
    #     # by_name = True ==> 같은 이름의 layer에 가중치 로딩
    #     model.load_weights('checkpoints/yolov1_weights.hdf5', by_name=True)
    #     print('complete loading yolov1_weights ')
    # else:
    #     print('no train history')

    if os.path.exists('checkpoints/tiny-yolov1.hdf5'):
        # by_name = True ==> 같은 이름의 layer에 가중치 로딩
        model.load_weights('checkpoints/tiny-yolov1.hdf5', by_name=True)
        print('complete loading yolov1_weights ')
    else:
        print('no train history')

    datasets_path = './VOCdevkit'  # dataset 존재하는 경로 위치

    # 데이터셋 제너레이터 생성
    train_generator = SequenceData(
        'train', datasets_path, input_shape, batch_size)
    validation_generator = SequenceData(
        'val', datasets_path, input_shape, batch_size)
    #print('====================================')

    model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[checkpoint, early_stopping],
        shuffle=False
    )
   # model.save_weights('yolov1_weights.hdf5')
    model.save_weights('mytiny-yolov1.hdf5')

if __name__ == '__main__':
    yolov1_main()
