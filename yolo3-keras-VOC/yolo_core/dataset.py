#import os
# 사용라이브러리 import
import cv2
import random
import numpy as np
import tensorflow as tf
from yolo_core.utils import *
from configuration import *

class Dataset():
    def __init__(self, dataset_type):
        # train/test annotation 파일, input_sizes, batch_size, data_aug 상태 설정
        self.annot_path  = TRAIN_ANNOT_PATH if dataset_type == 'train' else TEST_ANNOT_PATH
        # annot_path ==>  "./dataset/voc/voc2012_train.txt"
        self.input_sizes = TRAIN_INPUT_SIZE if dataset_type == 'train' else TEST_INPUT_SIZE
        # input_sizes ==> 416 x 416
        self.batch_size  = TRAIN_BATCH_SIZE if dataset_type == 'train' else TEST_BATCH_SIZE
        # 16 batch_size 를 --> 1 batch_size로 수정해 테스트중
        self.data_aug    = TRAIN_DATA_AUG   if dataset_type == 'train' else TEST_DATA_AUG
        # train 경우 data_aug = True

        self.strides = np.array(YOLO_STRIDES) # [8, 16, 32]
        self.classes = read_class_names(TRAIN_CLASSES)  # voc2012.names 사전타입으로 변환
        #print(self.classes)  {0: 'aeroplane', 1: 'bicycle', 2: 'bird',..... }
        self.num_classes = len(self.classes)
        #print('num_classes : ', self.num_classes)
        #print('YOLO ANchors : ', np.array(YOLO_ANCHORS))
        #print('YOLO ANchors.T : ', np.array(YOLO_ANCHORS).T)
        self.anchors = (np.array(YOLO_ANCHORS).T / self.strides).T
        #print('anchors : ', self.anchors)
        #print('anchors : ', self.anchors[0])
        self.anchor_per_scale = YOLO_ANCHOR_PER_SCALE  # 스케일 당 anchor 수 ==> 3
        self.max_bbox_per_scale = YOLO_MAX_BBOX_PER_SCALE  # 스케일 당 최대 GT bbox 개수 : 100
        self.annotations = self.load_annotations(dataset_type)  # 랜덤 loading
        #print('annotations ')
        #print(self.annotations)
        self.num_samples = len(self.annotations)
        #print('annotation len : ',self.num_samples)  # 148개
        # num_batchs ==> 1 epoch 동작을 위한 batch 길이(수)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0

    def load_annotations(self, dataset_type):
        final_annotations = []
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]

        #print('anntations : ', annotations)
        if dataset_type == 'train':
            # SIZE_TRAIN ==>  512*TRAIN_BATCH_SIZE
            if len(annotations) > SIZE_TRAIN: # annatations개수가 512*TRAIN_BATCH_SIZE 클 경우
                # SIZE_TRAIN = 512*TRAIN_BATCH_SIZE 개수 만큼만 랜덤 추출
                annotations = random.sample(annotations, SIZE_TRAIN)
        else:
            if len(annotations) > SIZE_TEST:
                annotations = random.sample(annotations, SIZE_TEST)

        #print(dataset_type, "length :", len(annotations))  # 148개

        np.random.shuffle(annotations) # annotations 정보를 랜덤으로 섞어줌

        for annotation in annotations:
            # fully parse annotations
            line = annotation.split()
            #print('line : ', line)
            image_path, index = "", 1
            for i, line_item in enumerate(line):
                # isnumeric() : 문자열이 숫자로 인식 되는지 체크
                if not line_item.replace(",", "").isnumeric():
                    #if image_path != "": image_path += " "
                    image_path += line_item  # '~~.jpg'
                else:
                    index = i
                    break
            # print('image_path : ', image_path)
            # if not os.path.exists(image_path):
            #     raise KeyError("%s does not exist ... " %image_path)
            if TRAIN_LOAD_IMAGES_TO_RAM:
                image = cv2.imread('./dataset/voc/' + image_path)
                # print('read image')
            else:
                image = ''
            # ['2008_000289.jpg', ['47,149,185,500,14'], array([[[  8,  77, 170],
            #         [ 12,  83, 174],.....]]] ) 정보 저장
            final_annotations.append([image_path, line[index:], image])

        #print('final_anno :', final_annotations)

        return final_annotations

    def __iter__(self):
        return self  # 현재 인스턴스를 반환

    # iterator 클래스 메서드
    def __next__(self):
        with tf.device('/cpu:0'):  # CPU 실행
            #self.input_sizes = random.choice([self.input_sizes])
            self.output_sizes = self.input_sizes // self.strides  # [52 26 13]

            batch_image = np.zeros((self.batch_size, self.input_sizes, self.input_sizes, 3), dtype=np.float32)
            #print(batch_image.shape) # (1, 416, 416, 3)

            batch_label_sbbox = np.zeros((self.batch_size, self.output_sizes[0], self.output_sizes[0],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
            batch_label_mbbox = np.zeros((self.batch_size, self.output_sizes[1], self.output_sizes[1],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
            batch_label_lbbox = np.zeros((self.batch_size, self.output_sizes[2], self.output_sizes[2],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
            #print('batch_label_lbbox shape : ', batch_label_lbbox.shape) # (1, 13, 13, 3, 25)
            batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            #print('batch_lbboxes shape : ', batch_lbboxes.shape) # (1, 100, 4)

            exceptions = False  # 잘못된 annotation dataset 에 대한 예외 처리
            num = 0
            # num_batchs  ==> num_samples / batch_size ==> 148 / 1
            if self.batch_count < self.num_batchs:  # 한 batch별 반복해 전체 dataset loading
                while num < self.batch_size:  # 한 batch size에 대한 annotation parsing
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples:  # index가 dataset 크기보다 클 경우 index 감소시켜 오류 해결
                        index -= self.num_samples
                    annotation = self.annotations[index]
                    #print('annotation : ', annotation)
                    # ['image_path정보', bboxes정보, 실제 rgb imagedata]
                    # 해당 index에 대한 image 증강 및 증강에 따른 bboxes 정보 파싱해 로딩
                    image, bboxes = self.parse_annotation(annotation)
                    try:
                        label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes)
                    except IndexError:
                        exceptions = True
                        print("IndexError, something wrong with")

                    batch_image[num, :, :, :] = image
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    #print('sbboxes shape : ', sbboxes.shape)
                    # [batch_size, box수, box좌표 ] = [box수, box좌표 ]
                    # batch 단위별 box수에 대한 좌표 저장(3차원배열에 2차원배열 저장)
                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes
                    num += 1

                if exceptions:
                    print('\n')
                    raise Exception("There were problems with dataset, I fixed them, now restart the training process.")

                self.batch_count += 1
                #print('batch cout : ', self.batch_count)
                #print('batch_sbboxes shape : ', batch_sbboxes.shape) # (1, 100, 4)
                #print('batch_label_sbbox shape : ', batch_label_sbbox.shape)  # (1, 52, 52, 3, 25)
                batch_smaller_target = batch_label_sbbox, batch_sbboxes
                batch_medium_target = batch_label_mbbox, batch_mbboxes
                batch_larger_target = batch_label_lbbox, batch_lbboxes

                # batch size 만큼 --> image_data, target 로 unpacking
                return batch_image, (batch_smaller_target, batch_medium_target, batch_larger_target)
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)
                print("iterator stop!! 1 epoch finish!! ")
                raise StopIteration  # iterator stop 을 위해 꼭 필요함

    def parse_annotation(self, annotation):
       # print('parse_annotation')
        #print(annotation)
        if TRAIN_LOAD_IMAGES_TO_RAM:   # ['image_path정보', bboxes정보, 실제 rgb imagedata]
            image_path = annotation[0]
            image = annotation[2]
        else:
            image_path = annotation[0]
            image = cv2.imread(image_path)

        bboxes = np.array([list(map(int, box.split(','))) for box in annotation[1]])
        #print('bboxes ', bboxes)
        if self.data_aug:  # 이미지 증강
            image, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_translate(np.copy(image), np.copy(bboxes))

        #print('data aug bboxs : ', bboxes)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 이미지 증강에 따른 image 및 GT boxes 조절
        image, bboxes = image_preprocess(np.copy(image), [self.input_sizes, self.input_sizes], np.copy(bboxes))
        #print('bboxes : ', bboxes) # [ 58  78 353 402  14] ==> GT boxes_좌표 , class구분 정보
        #print('image', image.shape)
        return image, bboxes

    def preprocess_true_boxes(self, bboxes):
        # [52 26 13] output_sizes 에 따른 labeling Data 저장 배열 선언
        label = [np.zeros((self.output_sizes[i], self.output_sizes[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)]
        # print('self.anchor_per_scale : ', self.anchor_per_scale )
        # print(' self.num_classes : ',  self.num_classes)
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        #print('bbboxes_xywh shape : ', np.zeros((self.max_bbox_per_scale, 4)).shape)
        bbox_count = np.zeros((3,))  # [0. 0. 0.]==>[8 scale box수, 16scale box수, 32scale box수]
        #print('src bboxes : ',bboxes)
        #[[51 170 327 323  19]
        # [148  67 287 176  14]
        #print('bboxes len : ', len(bboxes))
        for bbox in bboxes: # GT boxes 개수 만큼 반복
            bbox_coor = bbox[:4]  # gt boxes 좌표
            bbox_class_ind = bbox[4] # gt boxes 클래스 구분자

            # self.num_classes :  20
            # class 구분 label을 smoothing labeling화 ==> 모델의 일반화 성능을 높여줌
            # smoothing labeling 화 ==> 모델이 0.9 정도면 잘 예측했다 로 loss 산출 하기 위해
            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0  # 해당 클래스 위치만 1로 표현
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution
            # 소숫점의 과학적 표기법의 사용 억제
            # np.set_printoptions(suppress=True)
            # print(smooth_onehot)

            # axis = -1  :  가장 낮은 차원(1차원) 기준으로 병합
            # 박스좌표를 중심좌표와 너비,높이로 변환
            bbox_xywh = np.concatenate([(bbox_coor[2:]+bbox_coor[:2])*0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            #print(' bbox_xywh ' )  # bbox 중심좌표
            #print(bbox_xywh)
            #print(self.strides) # [ 8 16 32]
            # np.newaixs ==> np 행렬의 차원을 확장
            # arr : [1 2 3 4] ==> arr[:,np.newaxis] ==> [[1],
            # #                                          [2],
            # #                                          [3],
            # #                                          [4]]
            # arr : [1 2 3 4] ==> arr[np.newaxis, :] ==> [[1 2 3 4]]
            # print(self.strides[:,np.newaxis]) ==> [[ 8]
            #                                        [16]
            #                                        [32]]
            # 1.0 곱해서 float으로 바꿔줌
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]
            #print(bbox_xywh_scaled) #==> 3개의 scale로 변환된 좌표
            # [[32.25     12.6875   19.75     12.375]   # <== 8 scale 조정좌표(small object detection)
            #  [16.125     6.34375   9.875     6.1875]   # <== 16 scale 조정좌표(midium object detection)
            #  [8.0625      3.171875 4.9375  3.09375]]    # <== 32 scale 조정좌표(large object detection)

            iou = []
            exist_positive = False
            for i in range(3):
                # i가 0일 경우 => 8 scale조정좌표, i가 1일 경우 => 16 scale 조정좌표
                # i가 2일 경우 => 32 scale조정좌표
                # 각 stride별 GT box 중심좌표의 그리드 셀 기준 3개의 Prior Anchor 생성
                anchors_xywh = np.zeros((self.anchor_per_scale,4))
                # np.floor ==> 내림함수(주어진 숫자와 같은정수 또는 주어진 숫자보다 작으며 가장 가까운 정수 반환)
                # 아래코드 : GT boxes좌표 기준의 prior anchor의 중심좌표 계산
                anchors_xywh[:,0:2] = np.floor(bbox_xywh_scaled[i,0:2]).astype(np.int32)+0.5
                # 각 스케일의 다른 3개의 prior anchor 너비, 높이 적용
                anchors_xywh[:,2:4] = self.anchors[i]
                #print(" anchors_xywh shaep : ", anchors_xywh.shape) # (3, 4)
                #print(anchors_xywh)

                # for 3회 반복 하면서 GT boxes 와 prior anchor간 IOU 계산
                # 1. 8 scale GT boxes좌표와 3개의 8스케일 prior anchor간 IOU 계산 ==> 3개
                # 2. 16 scale GT boxes좌표와 3개의 16스케일 prior anchor간 IOU 계산 ==> 3개
                # 3. 32 scale GT boxes좌표와 3개의 32스케일 prior anchor간 IOU 계산  ==> 3개
                #print('bbox_xywh_scaled[i][np.newaxis,:] shape : ', bbox_xywh_scaled[i][np.newaxis,:].shape) # (1, 4)
                iou_scale = bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                #print('iou_scale : ', iou_scale)
                iou.append(iou_scale) # 총 9개 iou 저장
                iou_mask = iou_scale > 0.3
                #print('iou_mask')
                #print(iou_mask) # ==> [False False False] boolean
                # np.any(iou_mask) # iou_mask 값중 하나라도 True 이면 True 반환
                if np.any(iou_mask):  # iou가 0.3보다 클 경우에 해당 grid지점에 GT 데이터 정보 저장
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)
                    # i가 0일 경우 => 8 scale label, i가 1일 경우 => 16 scale label
                    # i가 2일 경우 => 32 scale label
                    # iou_mask ==> [False  True  True] 형태의 각 스케일별 3개
                    # iou_mask ==> [False  True  True]  boolean 배열로
                    # True 인 위치만 데이터 저장
                    label[i][yind, xind, iou_mask, : ] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh  # 원래 중심좌표
                    label[i][yind, xind, iou_mask, 4:5] = 1.0  # confidence score
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    # loss function 에서 배경에 대한 loss 밸런싱을 계산하기 위해
                    # GT bbox 정보만 별도 저장
                    # bbox_count[i] % self.max_bbox_per_scale
                    # ==> 나머지 값 반환하게 해서 100개 미만으로 저장
                    # 각 scale별 100개의 임시 bboxes 배열에 bbox정보 저장
                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    #print("bbox_ind : ", bbox_ind)
                    #print('iou가 높은 bbox_xywh : ', bbox_xywh)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            # prior anchor와 GT간 iou가 낮은 경우 iou가 가장큰 prior anchor를 best anchor로
            if not exist_positive:
                #print(np.array(iou).reshape(-1))  # 9개의 iou 값을 1차 배열로 반환
                #print(np.argmax(np.array(iou).reshape(-1),axis=-1)) # 가장 큰 iou 인덱스 반환
                best_maxiou_anchor_ind = np.argmax(np.array(iou).reshape(-1),axis=-1)
                # 어떤 스케일의 몇번째 anchor 인지를 찾음
                best_scale_detect = int(best_maxiou_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_maxiou_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_scale_detect, 0:2]).astype(np.int32)

                # 해당 스케일의 몇번쨰 anchor가 best 인지를 나타냄
                label[best_scale_detect][yind, xind, best_anchor, :] = 0
                label[best_scale_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_scale_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_scale_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_scale_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_scale_detect][bbox_ind, :4] = bbox_xywh
                #print('iou가 낮은 best anchor의 bbox_xywh : ', bbox_xywh)
                bbox_count[best_scale_detect] += 1

        #  8, 16, 32 stride 별 라벨 및 bboxes 정보를  분리해서 반환
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh # bboxes_xywh[0],bboxes_xywh[1],bboxes_xywh[2]
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes



    def random_horizontal_flip(self, image, bboxes):
        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0,2]] = w - bboxes[:, [2,0]]

        return image, bboxes

    def random_crop(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes

    def random_translate(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes

    def __len__(self):
        return self.num_batchs
