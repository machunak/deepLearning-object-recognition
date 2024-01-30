from Faster_RCNN.Vgg16Model import Vgg16Cls
from Faster_RCNN.RPN import RPNCls
from Faster_RCNN.RCNN import RCNNCls
from tensorflow.keras.models import load_model
from Faster_RCNN.ImagePreprocess import *
from Faster_RCNN.Anchors_IOU import AnchorsCls
from Faster_RCNN.ROI import ROICls
from Faster_RCNN.ROIPooling import ROIPoolingCls

import cv2
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf) #무한으로 출력합니다. (sys.maxsize 크기 만큼 출력

# 소숫점의 과학적 표기법의 사용 억제
np.set_printoptions(suppress=True)
class TestCls():
    def __init__(self, number_of_anchorboxes_per_anchor,
                 anchor_box_scales, anchor_box_ratios,
                 anchor_box_base_size, classes, roipool_size=(7,7), epochs=1):
        self.k = number_of_anchorboxes_per_anchor
        self.anchor_box_scales = anchor_box_scales
        self.anchor_box_ratios = anchor_box_ratios
        self.anchor_box_base_size = anchor_box_base_size
        self.classes = classes
        self.roipool_size = roipool_size
        self.epochs = epochs
        self.PreVgg16 = Vgg16Cls()  # vgg16 모델 전이학습 (17layer)활용
        self.vgg_layer = 17
        model_folder = "./Trained_Models/"
        vgg16_featuremap_ChannelNum = self.PreVgg16.get_featuremap_ChannelNum()
        #l1smooth_loss = l1smooth()
        #self.rpn_model = load_model(model_folder + "rpn_model.h5", custom_objects={"l1smooth":l1smooth_loss})
        self.rpn_model = load_model(model_folder + "rpn_model.h5")
        #number_of_classes = len(self.classes)
        #self.rcnn_model = load_model(model_folder + "rcnn_model.h5", custom_objects={"l1smooth":l1smooth_loss})
        self.rcnn_model = load_model(model_folder + "rcnn_model.h5")

    def test_Update_image(self, image,  image_class_names):
        print("test image size and bbox info update!!")
        # image ==> 원본 이미지 shape : (400, 600, 3)
        # image_bounding_boxes ==> 이미지 bbox 좌표 정보

        # 원본 이미지의 작은 면이 600픽셀이 되도록 이미지 사이즈 조절
        # Vgg16 의 FC layer를 제외한 17 layer만 사용함으로 다른 사이즈의 이미지를
        # 입력으로 사용할 수 있으나 작은 면이 600픽셀이 되도록 이미지 사이즈를
        # 고정함으로서 더 디테일한 feature맵을 얻을 수 있어 성능이 향샹되는 경향이 있음
        # 예) (400, 600, 3) --> (1, 600, 800, 3) 되도록 이미지 사이즈 조절
        # 사이즈 조절에 따른 bounding 박스 정보 수정
        self.new_image, self.new_image_height, self.new_image_width =\
            test_prepare_image_bboxes(image)

        print('new_image shape : ', self.new_image.shape)

        self.image_class_names = image_class_names  # ['car']

        # new image 와  new GT box 출력
        #new_image_GT_bboxes_display(self.new_image, self.new_image_bounding_boxes)

        # 갱신된 이미지와 GT boxes 정보를 활용 학습 시작
        self.Test_start()

    def Test_start(self):
        print('test start!!')

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
                             None,feature_map_height,
                             feature_map_width,
                             downscale,
                             self.k,   # 9
                             self.anchor_box_scales,
                             self.anchor_box_ratios,
                             self.anchor_box_base_size ) # 132


        # 9개 다른 모양의 앵커 생성
        anchors.Create_anchor()

        rpn_pred_classes, rpn_pred_deltas = self.rpn_model.predict(feature_map)
        # 앵커 박스와 RPN 예측 출력값의 class score와 regressor deltas 활용
        # regions of interest(ROI) 추출 작업 수행 ( ROI 획득 )
        # ROI class 객체 생성
        r_obj = ROICls(anchors, rpn_pred_classes, rpn_pred_deltas)
        # ROI boxes 와 추정치 획득
        roi_boxes, roi_boxes_proba = r_obj.get_proposals()
        # anchors.Anchor_box_display(roi_boxes, self.new_image)

        # roi_boxes_proba : 추정치 구분자를 바탕으로
        # 전경 + 배경 ==> 256개 batch 데이터 획득
        roi_boxes_batch = r_obj.get_roi_batch(roi_boxes, roi_boxes_proba)
        # print(roi_boxes_batch)
        print('roi_boxes_batch.shape : ', roi_boxes_batch.shape)
        # anchors.Anchor_box_display(roi_boxes_batch[:50], self.new_image)

        # roi Pooling layer 구성
        # feature_map 기준으로 pooling 적용해야 함으로
        # roi_boxes_batch 좌표값 에 downscale(16) 적용
        roi_boxes_batch_downscaled = roi_boxes_batch / downscale
        # print(roi_boxes_batch_downscaled)
        # print("feature map shape: ", feature_map.shape)

        # roipooing 객체 생성
        roipooling = ROIPoolingCls(mode='tf', pool_size=self.roipool_size)  # (7,7)
        # pooling 작업 된 roi boxes batch 획득
        pooled_roi_boxes_batch = roipooling.get_Pooled_roi_batch(feature_map, roi_boxes_batch_downscaled)
        # (256, 7, 7, 512)
        print('pooled roi boxes batch shape : ', pooled_roi_boxes_batch.shape)

        rcnn_classifier, rcnn_regressor = self.rcnn_model.predict(pooled_roi_boxes_batch)


        print("최종 분류 및 bbox 예측")
        print("rcnn_classifier len : ", len(rcnn_classifier) )
        counter = 0
        foregroundScorelist = []
        fgRegBoxlist = []
        for i in range(len(rcnn_classifier)):
            #print(rcnn_classifier[i])
            class_idx = np.argmax(rcnn_classifier[i])  #  class_idx ==> 1 이 전경인 roi
            if class_idx == 0:  # 높은값 인덱스가 1 이 전경 roi
                continue
            counter += 1
            # print("found box: ",counter)
            dx, dy, dw, dh = rcnn_regressor[i, 4 * (class_idx - 1):4 * (class_idx - 1) + 4]

            # roi boxes 기준으로 좌표 복원
            # GT boxes와 roi boxes 와의 차이를 RCNN 타깃으로 설정했기 떄문에
            # roi boxes 기준으로 복원한 좌표는 GT boxes와 유사
            xmin, ymin, xmax, ymax = anchors.decoding_bbox_by_deltas(roi_boxes_batch[i], [dx, dy, dw, dh])
            # print(xmin, ymin, xmax, ymax)

            #if (rcnn_classifier[i][1] > 0.43):  # 0.42  , 0.4005
                #   maxbox = rcnn_classifier[i][1]
            foregroundScorelist.append(rcnn_classifier[i][1])
            fgRegBoxlist.append([xmin, ymin, xmax, ymax])
            # _, ax = plt.subplots(1)
            # ax.imshow(np.squeeze(self.image))
            # ax.add_patch(patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, facecolor='none', edgecolor='r', linewidth=1))
            # plt.show()
        print('Foreground counter : ', counter)
        #print(np.array(findboxlist))
        if counter > 0:
            maxidx = np.argmax(foregroundScorelist)  # 전정추정치 가장큰 인덱스 추출
            print('maxidx : ', maxidx)
            print('score :' , foregroundScorelist[maxidx])
            box = fgRegBoxlist[maxidx]
            x_min = int(box[0])
            y_min = int(box[1])
            x_max = int(box[2])
            y_max = int(box[3])
            self.new_image = np.squeeze(self.new_image)
            cv2.rectangle(self.new_image, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=3)
            # OpenCV가 이미지를 read할떄 RGB 형태가 아닌 BGR 형태로 로딩하기 때문에
            # 색감이 원본 이미지와 다르게 나타남
            # BGR을 RGB로 바꿔주어야 한다(BGR2RGB)
            self.new_image = cv2.cvtColor(self.new_image, cv2.COLOR_BGR2RGB)
            plt.imshow(self.new_image) # 이미지 랜더링
            plt.show() # 이미지 출력




