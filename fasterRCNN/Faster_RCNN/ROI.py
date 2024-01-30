import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import pandas as pd
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width',1000)
pd.set_option('max_colwidth', 1000)
#pd.set_option('display.colheader_justify','center')  # 컬럼 중앙 출력
pd.set_option('display.float_format', '{:.3f}'.format) #  float 형식 소숫점 3자리 표현


# 소숫점 이하 3자리까지 출력
np.set_printoptions(precision=5)
np.set_printoptions(threshold=np.inf) #무한으로 출력합니다. (sys.maxsize 크기 만큼 출력

class ROICls():
    def __init__(self, anchors_obj, rpn_pred_classes, rpn_pred_deltas):
        print('ROI class init!!')
        self.anchors = anchors_obj
        self.classes = rpn_pred_classes
        self.deltas = rpn_pred_deltas

    def get_proposals(self):
        print('get proposals')
        # rpn deltas 값을 앵커 boxes에 매칭하여
        # proposal boxes에 복원 좌표저장 및  probabilities에 전경/배경 추정치 저장
        proposal_boxes = []  # proposal anchor 좌표 list
        probabilities = []  # 전경 인지 배경인지 구별 정보

        #anchorlist = []
        for anchor_y in range(0, self.anchors.image_height - 16, self.anchors.downscale):  # 원본 이미지 size를 y축으로 16 down스케일로 이동
            for anchor_x in range(0, self.anchors.image_width - 16, self.anchors.downscale):  # # 원본 이미지 size를 x축으로 16 down스케일로 이동
                for anchorShapebox_idx in range(len(self.anchors.anchor_shape_9_boxes)):  # 9 anchorShapebox
                    # 이미지 x축(가로) 사이즈 / 16스케일 ==> 예) 56 ( 0 ~ 55 인덱스 )
                    # 이미지 y축(세로) 사이지 / 16스케일 ==> 예) 37 ( 0 ~ 36 인덱스 )
                    # 9 anchorShapebox
                    # ==>  56 * 37 * 9 개 = 18648 개의 앵커 좌표 리스트 생성
                    anchor_oneitem_coordinate = self.anchors.anchor_shape_9_boxes[anchorShapebox_idx]
                    #  Anchor boxes 의 좌표 xmin, ymin 은 16 scale 절반인 8씩 더 하는
                    # 예) 앵커_X+8, 앵커_Y 인  (408, 400) 를 기준으로 좌표 값이 갱신 되어야 함
                    anchor_xy_adjust = [anchor_x + 8, anchor_y + 8]  # 앵커 위치 좌표 조정
                    # 1차원배열의 concatenate()은 axis=0만 존재하며 두 1차원배열이 그냥 합쳐짐
                    # 앵커 위치 조정 좌표를 앵커모양 좌표(xmin,ymin,xmax,ymax)에 반영하기위해 4개로 확장
                    anchor_xy_adjust_point = np.concatenate((anchor_xy_adjust, anchor_xy_adjust))  # [ 8 8 8 8]
                    # print(anchor_xy_adjust_point)
                    # 최종적으로 앵커모양 좌표에 앵커 위치 조정 좌표 반영하여 앵커박스 좌표 정보 갱신
                    anchor_box_coord_updated = anchor_oneitem_coordinate + anchor_xy_adjust_point

                    # 생성된 총 18648앵커중 앵커 좌표가 이미지 크기를 벗어나면 아래 코드로 생략(제거)
                    # ==> 이미지 크기를 벗어난 앵커 생략(제거)된 앵커 개수는 : 7326
                    if anchor_box_coord_updated[0] < 0 \
                            or anchor_box_coord_updated[1] < 0 \
                            or anchor_box_coord_updated[2] >= self.anchors.image_width \
                            or anchor_box_coord_updated[3] >= self.anchors.image_height:
                        continue

                    #anchorlist.append(anchor_box_coord_updated)
                    # feature map 기준으로 rpn 예측 차이(deltas) 값 복원 해야 함으로
                    # ay, ax 는 anchor_y 와  anchor_x를 downscale 적용해 활용
                    ay = anchor_y // self.anchors.downscale
                    ax = anchor_x // self.anchors.downscale

                    # 이미지 크기를 벗어나지 않은 7326개 앵커 box 기반해서
                    # rpn 예측 deltas 값의 ROI bbox 좌표 복원
                    # GT boxes 와의 차이를 rpn 타깃으로 주었기 떄문에
                    # loss가 가장 작은 rpn predict deltas는 GT boxes와 유사한 ROI boxes일 것음
                    # deltas ==> 0:4, 4:8 , 8:12,........ 줘야 함
                    xmin, ymin, xmax, ymax = self.anchors.decoding_bbox_by_deltas( anchor_box_coord_updated,
                                                                                   self.deltas[0,ay,ax,
                                                                                   4*anchorShapebox_idx : 4*anchorShapebox_idx+4] )

                    # decoding bboxes Cliping
                    # 만약 x,y 값이 음의값등.. 이미지 경계를 벗어나면 x, y 값을 자름
                    xmin = np.maximum(0, xmin)  # 음의 값이면 0, 0보다 큰값이면 xmin
                    ymin = np.maximum(0, ymin)  # 음의 값이면 0, 0보다 큰값이면 ymin
                    xmax = np.minimum(self.anchors.image_width, xmax)  # 이미지 너비 또는 이미지 너비 보다 작은 값 선택
                    ymax = np.minimum(self.anchors.image_height, ymax)  # 이미지 높이 또는 이미지 높이 보다 작은 값 선택
                    # print(xmin, ymin, xmax, ymax)
                    # 복원된 ROI bbox 좌표가 반대이면 건너띔
                    if ((xmin - xmax) >= 0) or ((ymin - ymax) >= 0):
                        continue

                    #anchorlist.append([xmin, ymin, xmax, ymax])

                    # 전경 타깃 ==> [1 0], 배경 타깃 ==> [0 1]

                    # 추후 roi_batch 데이터 생성을 위해
                    # 전경/배경 구별된 추정치 와 구분자 별도 저장
                    # 전경 추정 데이터
                    foreground_prob = self.classes[0, ay, ax, 2*anchorShapebox_idx]
                    # 배경 추정 데이터
                    background_prob = self.classes[0, ay, ax, 2*anchorShapebox_idx+1]
                    # 전경 추정데이터가 배경 추정 데이터 보다 크다면
                    # 해당 영역은 1을 갖는 전경으로  아니면  배경(0)으로 저장
                    probabiliy, c = (foreground_prob, 1) if foreground_prob >= background_prob else (background_prob, 0)
                    proposal_boxes.append([xmin, ymin, xmax, ymax]) # ROI 복원좌표
                    probabilities.append([probabiliy, c]) # [추정치 , 구분자]

        # probabilities(추정치)
        # 전경일 경우 [ 0.67 , 1 ]
        # 배경일 경우 [ 0.89 , 0 ]
        proposal_boxes = np.array(proposal_boxes) # numpy배열로 변환
        probabilities = np.array(probabilities) # numpy 배열로 변환
        #print(' probabilities len : ', len(probabilities))
        # print(np.array(anchorlist[:10]))
        # print('anchorlist len : ', len(anchorlist))
        # self.anchors.Anchor_box_display(anchorlist[:10], self.anchors.image)
        print('started non max suppession')
        # nms 동작 후 겹치지 않은 앵커 박스 인덱스 반환 ( 전경추정 앵커 + 배경추정 앵커 )
        # iou_threshold = 0.5 기반으로 동일 object에 대한 Detection 제거
        # nms동작 : 주어진 Box들 중 가장 높은 Score를 가진 Box를 선택
        # 선택된 Box와 나머지 Box들간의 iou(interaction of union)을 계산하고 특정값(iou threshold) 이상이면 제거한다.
        # (동일한 Object에 대한 Detection이기 때문에 겹치는 부분이 많을 것이라는 생각)
        # 특정한 Box의 숫자가 남을 때까지나 더이상 선택할 Box가 없을 때까지 위의 과정을 반복
        # 제거되고 남은 roi proposals 인덱스 반환
        selected_indices = tf.image.non_max_suppression(proposal_boxes, probabilities[:, 0], max_output_size=2000)
        #print('selected_indices : ', selected_indices)  # 유사 앵커박스 제거되고 남은 인덱스
        roi_boxes = np.take(proposal_boxes, selected_indices, axis=0)
        roi_probabilities = np.take(probabilities, selected_indices, axis=0)
        print('NMS 이후 :', len(roi_boxes))
        #self.anchors.Anchor_box_display(roi_boxes, self.anchors.image)
        # print(roi_boxes)
        # print(roi_probabilities)
        return roi_boxes, roi_probabilities

    def get_roi_batch(self, roi_boxes, roi_probabilities, batch_size=256, ratio_of_foregrounds=0.5):
        roi_boxes_proba = np.column_stack((roi_boxes, roi_probabilities))
        #print(roi_boxes_proba)  # [배경추정치, 0] , [전경추정치, 1]
        roi_boxes_proba_df = pd.DataFrame(roi_boxes_proba,columns=['xmin','ymin','xmax','ymax','prob','c'])
        #print(roi_boxes_proba_df)  # NMS 이후 383 개
        # 배경 개수
        foreground_len = len(roi_boxes_proba_df.loc[roi_boxes_proba_df['c'] == 1].index)
        #print(foreground_len) 50

        # roi_batch 비율
        f = int(batch_size * ratio_of_foregrounds)  # 128

        if foreground_len > f:
            # 전경 선택
            foreground_sel = roi_boxes_proba_df.loc[roi_boxes_proba_df['c'] == 1]
            # 전경 선택 중에 추정치가 높은 순으로 정렬
            foreground_sorted = foreground_sel.sort_values(by='prob', ascending=False)
            # 추정치 높은 128개 전경 roi_boxes 좌표만 추출
            foreground_boxes = foreground_sorted.iloc[0:f, :4]

            # 배경 선택
            background_sel = roi_boxes_proba_df.loc[roi_boxes_proba_df['c'] == 0]
            # 배경 선택 중에 추정치가 높은 순으로 정렬
            background_sorted = background_sel.sort_values(by='prob', ascending=False)
            b = int(batch_size * (1 - ratio_of_foregrounds))  # 128
            background_boxes = background_sorted.iloc[0:b, :4]

            # 전경 + 배경 합쳐서 roi_boxes 값만 추출(np.array)
            roi_boxes_batch = pd.concat([foreground_boxes, background_boxes]).values
            #print(len(roi_boxes_batch))

        else:
            # 전경 모두 사용 과 남어지는 추정치 높은 배경기준으로 배경 더해서 총 256개
            # 전경 선택
            foreground_sel = roi_boxes_proba_df.loc[roi_boxes_proba_df['c'] == 1]
            # roi_boxes 좌표만 추출
            foreground_boxes = foreground_sel.iloc[:,0:4]

            # 배경 선택
            background_sel = roi_boxes_proba_df.loc[roi_boxes_proba_df['c'] == 0]
            # 배경 선택 중에 추정치가 높은 순으로 정렬
            background_sorted = background_sel.sort_values(by='prob', ascending=False)
            # 256 배치사이즈 중 전경 길이를 뺀 남어지는 배경으로
            b = batch_size - foreground_len
            background_boxes = background_sorted.iloc[0:b, :4]

            # 전경 + 배경 합쳐서 roi_boxes 값만 추출(np.array)
            roi_boxes_batch = pd.concat([foreground_boxes, background_boxes]).values
            #print(len(roi_boxes_batch))
            #print(roi_boxes_batch)

        return roi_boxes_batch




