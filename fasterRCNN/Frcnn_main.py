from Faster_RCNN.Menagement import MenagementCls
from Faster_RCNN.Train import TrainCls
from Faster_RCNN.Test import TestCls

# train / test 구별
# Oper_phase = 'training'
Oper_phase = 'testing'

# Dataset 위치 경로
training_dataset_folder = './Stanford_Dataset/cars_train/'
testing_dataset_folder = './Stanford_Dataset/cars_test/'

# 앵커박스 생성 스케일 지정
anchorbox_scales = [1, 2, 3]
anchorbox_ratios = [0.5, 1, 2]
anchorbox_base_size = 132  # 적절한 크기의 앵커 박스 생성하기 위해 기본값 설정

# 총 앵커 당 앵커박스 생성 수 ==> 9개
NumberOfanchorboxes_per_anchor = len(anchorbox_scales) * len(anchorbox_ratios)

# 학습할 이미지 로딩 기능의 MenagementCls 객체 생성
Menager = MenagementCls()

if Oper_phase == 'training':
    traingObj = TrainCls(NumberOfanchorboxes_per_anchor, anchorbox_scales,
                         anchorbox_ratios,
                         anchorbox_base_size)

    # train객체 와 train 이미지 데이터셋 전달하여 학습할 준비
    Menager.start_imgload_training(traingObj, training_dataset_folder)

elif Oper_phase == 'testing':
    classes = {'car':1}
    testingObj = TestCls(NumberOfanchorboxes_per_anchor, anchorbox_scales,
                         anchorbox_ratios,
                         anchorbox_base_size,
                         classes)

    # testing객체 와 test 이미지 데이터셋 전달하여 test dataset 예측
    Menager.start_imgload_testing(testingObj, testing_dataset_folder)

