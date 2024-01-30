from Faster_RCNN.Vgg16Model import Vgg16Cls
from Faster_RCNN.RPN import RPNCls
from Faster_RCNN.RCNN import RCNNCls
from tensorflow.keras.models import load_model
from Faster_RCNN.ImagePreprocess import *
from Faster_RCNN.Anchors_IOU import AnchorsCls
import cv2
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf) #무한으로 출력합니다. (sys.maxsize 크기 만큼 출력

# 소숫점의 과학적 표기법의 사용 억제
np.set_printoptions(suppress=True)

class TrainCls():
    def __init__(self, NumberOfanchorboxes,  # 9
                 anchorbox_scales,  # [1, 2, 3]
                 anchorbox_ratios,  #  [0.5, 1, 2]
                 anchorbox_base_size,  # 132
                 roipool_size=(7,7), epoches = 1):
        print("TrainCls!!")
        self.anchor_k = NumberOfanchorboxes  # 매개변수를 Trainclass 멤버변수 화
        self.anchor_scales = anchorbox_scales
        self.anchor_ratios = anchorbox_ratios
        self.anchor_base_size = anchorbox_base_size
        self.classes = {'car':1}  # rcnn 모듈의 출력층 개체 클래스 구별 위한 데이터
        self.roipool_size = roipool_size
        self.epoches = epoches
        self.PreVgg16 = Vgg16Cls()  # vgg16 모델 전이학습 (17layer)활용
        #self.PreVgg16.get_summery()  # Vgg16 상단 FC층 제외한 모델 활용
        self.rpn_model_dir = './Trained_Models/rpn_model.h5' # rpn모델 저장 경로
        self.rcnn_model_dir = './Trained_Models/rcnn_model.h5' # rcnn모델 저장 경로

        # Vgg16 17layer의 block5_conv3 layer outout shape에 channels 획득

        vgg16_featuremap_ChannelNum = self.PreVgg16.get_featuremap_ChannelNum()
        print('vgg16_featuremap_ChannelNum : ', vgg16_featuremap_ChannelNum) # 512

        # RPN / RCNN 모델 준비 및 생성
        # RPN 모델 입력 ==> featuremap
        # featuremap의 1 픽셀 좌표(앵커) 당 앵커박스 수(9개) 만큼의
        # 이미지 내부의 개체 유무 분류 및 해당 개체 경계상자(x,y,w,h)회귀 학습
        # 따라서, RPN 모델 생성하기 위해 앵커수 와 특징맵 채널수가 필요함

        # 기존 저장된 RPN모델 있을 경우 로딩
        # 없을 경우 새로운 RPN 모델 생성
        try:
            self.rpn_model = load_model(self.rpn_model_dir)
            print('existing rpn model loading!!')
        except:
            print('new rpn model create!!')
            # anchor_k : 9 , vgg16_featuremap_ChannelNum : 512  ==>전달
            self.rpn_model = RPNCls().create_model(self.anchor_k, vgg16_featuremap_ChannelNum)

        # # rcnn detector 설계
        # NumberOfclasses = len(self.classes) # 1
        # # Faster_RCNN Detector 구현
        # # ROI Pooling 작업된 ROI Pooled Shape => 7, 7, 512
        # # 기존 저장된 RCNN모델 있을 경우 로딩
        # # 없을 경우 새로운 RCNN모델 생성
        # self.rcnn_model = RCNNCls(NumberOfclasses, input_shape=(*self.roipool_size, vgg16_featuremap_ChannelNum))
        # try:
        #     self.rcnn_model = load_model(self.rcnn_model_dir)
        #     print('existing rcnn model loading!!')
        # except:
        #     print('new rcnn model create!!')
        #     self.rcnn_model = self.rcnn_model.create_model()

    def Update_image(self, image, image_bounding_boxes, image_class_names):
        print("image size and bbox info update!!")
        # image ==> 원본 이미지 shape : (400, 600, 3)
        # image_bounding_boxes ==> 이미지 bbox 좌표 정보

        # 원본 이미지의 작은 면이 600픽셀이 되도록 이미지 사이즈 조절
        # Vgg16 의 FC layer를 제외한 17 layer만 사용함으로 다른 사이즈의 이미지를
        # 입력으로 사용할 수 있으나 작은 면이 600픽셀이 되도록 이미지 사이즈를
        # 고정함으로서 더 디테일한 feature맵을 얻을 수 있어 성능이 향샹되는 경향이 있음
        # 예) (400, 600, 3) --> (1, 600, 800, 3) 되도록 이미지 사이즈 조절
        # 사이즈 조절에 따른 bounding 박스 정보 수정
        self.new_image, self.new_image_height, self.new_image_width,self.new_image_bounding_boxes =\
            prepare_image_bboxes(image, image_bounding_boxes)

        print('new_image bboxes : ', self.new_image_bounding_boxes)
        print('new_image shape : ', self.new_image.shape)

        self.image_class_names = image_class_names

        # new image 와  new GT box 출력
        #new_image_GT_bboxes_display(self.new_image, self.new_image_bounding_boxes)

        # 갱신된 이미지와 GT boxes 정보를 활용 학습 시작
        self.Train_start()

    def Train_start(self):
        print('train start!!')

        # vgg16 모델 활용한 특징맵 추출
        # updated new 이미지를 입력으로
        # block5_conv3 layer(17 layer) of vgg16 까지만 활용하여 특징맵 추출
        feature_map, feature_map_shape = self.PreVgg16.get_feature_map(self.new_image)
        #print('Train feaure_map_shape : ', feature_map.shape) # (1, 37, 56, 512)

        # RPN 학습을 위해 앵커 생성
        # ==> feature_map 사이즈(height, width) 기반으로 앵커 생성 함으로
        # feature_map height, width 백업
        feature_map_height = feature_map_shape[-3]
        feature_map_width = feature_map_shape[-2]
        #print(feature_map_height, feature_map_width)
        print('====================== 특징맵 추출 완료 ===================')

        # 이미지 위 앵커 출력 코드
        #image_anchor_point_display(self.new_image, self.new_image_height, self.new_image_width)

        # 원본 이미지가 feature map 사이즈로 나눈 비율  만큼
        # 다운스케일 됨에 따라 다운스케일을 계산하여
        # 위 image_anchor_point_display() 함수 처럼 anchor 출력 좌표를 구함
        # anchor 좌표 당  9개의 앵커를 생성 해야 함

        downscale = self.new_image_height // feature_map_height
        print('downscale : ', downscale)
        # 예) downscale == 16인 경우
        # 16비율 만큼 축소된 위치의 한 픽셀좌표가
        # feature_map 기준 한 픽셀 좌표가 되며 해당 픽셀 좌표 당 앵커 9개를
        # 생성 및 앵커 타깃데이터를 생성하여  RPN에서 매칭 학습한 후 ROI 를 추출 해야함

        # Anchor 생성 및 타깃 생성을 위한 Anchors 객체 생성
        anchors = AnchorsCls(self.new_image,self.new_image_height,self.new_image_width,
                             self.new_image_bounding_boxes,feature_map_height,
                             feature_map_width,
                             downscale,
                             self.anchor_k,   # 9
                             self.anchor_scales,
                             self.anchor_ratios,
                             self.anchor_base_size ) # 132

        # 앵커 생성 및 앵커 타깃데이터 획득

        # RPN 모듈 out shape  ==> RPN_classes_layer : (1, 37, 56, 18) , RPN_regressor_layer : (1, 37, 56, 36)
        # 임으로  앵커 타깃 shape 도 동일하게
        # RPN_target_classes : (1, 37, 56, 18) , RPN_target_regressor : (1, 37, 56, 36)
        # 로 생성 해줘야 함
        rpn_target_classes, rpn_target_regressor = anchors.get_rpn_anchor_targetdata()
        print('rpn target create finished!!')
        print(rpn_target_classes.shape, rpn_target_regressor.shape)

        # RPN 모듈 학습 진행
        self.rpn_model.fit(feature_map, [rpn_target_classes, rpn_target_regressor],epochs=self.epoches)

        # RPN => predict(예측) regions of interest(ROI)
        rpn_pred_classes, rpn_pred_deltas = self.rpn_model.predict(feature_map)
        print('rpn_pred_classes : ', rpn_pred_classes[0][25][25])  # (1, 37, 56, 18)
        print('rpn_pred_deltas : ', rpn_pred_deltas[0][25][25])  # (1, 37, 56, 36)

        # 위치별 앵커 박스와 RPN 예측 반환한 class score와 boundingbox regressor 활용
        # regions of interest(ROI)의 region proposals 추출 작업 수행









