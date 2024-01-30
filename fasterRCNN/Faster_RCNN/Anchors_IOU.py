import numpy as np
import matplotlib.pyplot as plt
import cv2

class AnchorsCls():
    def __init__(self, image, image_height, image_width,
                 image_bounding_boxes,
                 feature_map_height,feature_map_width,
                 downscale,
                 anchor_k,
                 anchor_box_scales, anchor_box_ratios,
                 anchor_box_base_size):

        self.image = image
        self.image_height = image_height
        self.image_width = image_width
        self.image_bounding_boxes = image_bounding_boxes
        self.feature_map_height = feature_map_height
        self.feature_map_width = feature_map_width
        self.downscale = downscale
        self.anchor_k = anchor_k
        self.anchor_box_scales = anchor_box_scales
        self.anchor_box_ratios = anchor_box_ratios
        self.anchor_box_base_size = anchor_box_base_size
        print("Anchorscls __init__finish")

    def Create_anchor(self):  # 9개의 다른 모양의 앵커 생성
        self.anchor_shape_9_boxes = self.Create_anchor_Shape_by_ScaleRatio()

    def get_rpn_anchor_targetdata(self):
        print('get_rpn_anchor_targetdata')
        # feature_map 넓이/크기 기반 타깃 shape 생성
        anchor_rpn_classes_target = np.zeros((1,self.feature_map_height,self.feature_map_width, 2*self.anchor_k))
        anchor_rpn_regressors_target = np.zeros((1,self.feature_map_height,self.feature_map_width, 4*self.anchor_k))
        # (1, 37, 56, 18) (1, 37, 56, 36)
        #print(anchor_rpn_classes_target.shape, anchor_rpn_regressors_target.shape)

        # 앵커(중심좌표)에 적용할 9개의 앵커 모양 생성
        self.Create_anchor()
        #print(self.anchor_shape_9_boxes)
        # 다운 스케일 이미지 좌표를 따라 이동 하면서 좌표 센터에 따른 9개의 앵커 좌표
        # 조정하고 앵커가 이미지 크기를 벗어나면 해당 앵커 무시 함
        # 다운 스케일 이미지 좌표를 따라 이동 하면서  9개의 앵커를
        # GT box와 IOU를 비교 계산
        # IOU > 0.5 이상 인 경우 클래스 타깃 데이터는 1, 0 로 생성
        # IOU < 0.1 이하 인 경우 배경으로서 클래스 타깃 데이터는 0, 1 로 생성
        # 0.1 < iou < 0.5 인 앵커박스는 학습에 사용하지 않음

        anchor_IOU_high_list = []
        for anchor_y in range(0, self.image_height-16, self.downscale):  # 원본 이미지 size를 y축으로 16 down스케일로 이동
            for anchor_x in range(0, self.image_width-16, self.downscale): #  # 원본 이미지 size를 x축으로 16 down스케일로 이동
                for anchorShapebox_idx in range(len(self.anchor_shape_9_boxes)): # 9 anchorShapebox
                    # 이미지 x축(가로) 사이즈 / 16스케일 ==> 예) 56 ( 0 ~ 55 인덱스 )
                    # 이미지 y축(세로) 사이지 / 16스케일 ==> 예) 37 ( 0 ~ 36 인덱스 )
                    # 9 anchorShapebox
                    # ==>  56 * 37 * 9 개 = 18648 개의 앵커 좌표 리스트 생성
                    anchor_oneitem_coordinate = self.anchor_shape_9_boxes[anchorShapebox_idx]
                    #  Anchor boxes 의 좌표 xmin, ymin 은 16 scale 절반인 8씩 더 하는
                    # 예) 앵커_X+8, 앵커_Y 인  (408, 400) 를 기준으로 좌표 값이 갱신 되어야 함
                    anchor_xy_adjust = [anchor_x + 8 , anchor_y + 8] # 앵커 위치 좌표 조정
                    # 1차원배열의 concatenate()은 axis=0만 존재하며 두 1차원배열이 그냥 합쳐짐
                    # 앵커 위치 조정 좌표를 앵커모양 좌표(xmin,ymin,xmax,ymax)에 반영하기위해 4개로 확장
                    anchor_xy_adjust_point = np.concatenate((anchor_xy_adjust, anchor_xy_adjust))  # [ 8 8 8 8]
                    #print(anchor_xy_adjust_point)
                    # 최종적으로 앵커모양 좌표에 앵커 위치 조정 좌표 반영하여 앵커박스 좌표 정보 갱신
                    anchor_box_coord_updated = anchor_oneitem_coordinate + anchor_xy_adjust_point


                    # 생성된 총 18648앵커중 앵커 좌표가 이미지 크기를 벗어나면 아래 코드 생략(제거)
                    # ==> 이미지 크기를 벗어난 앵커 생략(제거)된 앵커 개수는 : 7326
                    if anchor_box_coord_updated[0] < 0 \
                            or anchor_box_coord_updated[1] < 0 \
                            or anchor_box_coord_updated[2] >= self.image_width \
                            or anchor_box_coord_updated[3] >= self.image_height:
                        continue

                    # Ground True image bounding boxes(이미지에 실제체크된 bounding box) 정보 로딩 후
                    # 생성된 해당 앵커박스와 iou(intersection Over Union) 계산 ==>2개의 영역 겹침 정도 계산
                    for GT_bounding_box in self.image_bounding_boxes:
                        # 생성된 해당 앵커박스와 GT bounding box 간의 IOU 계산
                        #print( anchor_box_coord_updated, GT_bounding_box)
                        iou = self.iou_calculation(anchor_box_coord_updated, GT_bounding_box)

                        # 계산된 IOU값 활용 앵커박스 타깃 라벨링
                        # iou >= 0.5 경우 전경 = 1, 배경 = 0   으로 해서 foreground(전경)으로 라벨링
                        # iou <= 0.1 경우 전경 = 0, 배경 = 1   으로 해서 background(배경)으로 라벨링
                        # 0.1 < iou < 0.5 경우 전경 = 0, 배경 = 0 으로 해서 학습 제외

                        # feature map shape 이 (1, 37, 56, ..) 경우
                        ay = anchor_y // self.downscale  # ay ==> 0~36 까지 인덱스 설정 토록
                        ax = anchor_x // self.downscale  # ax ==> 0~55 까지 인덱스 설정 토록
                        if iou >= 0.5: # 개체가 있을 확률이 높은 앵커박스
                            # feature map shape 이 (1, 37, 56, ..) 경우
                            # [0,0,0,0],[0,0,0,1]....[0,0,0,16],[0,0,0,17]
                            # [0,36,55,0]~~~[0,36,55,17] 까지  총 2*k = 18개 인덱스 증가 하도록 설정
                            # classed 라벨링 : 개체가 있을 확률이 높은 앵커는(전경) [1, 0] 으로 라벨링
                            anchor_rpn_classes_target[0, ay, ax, anchorShapebox_idx*2] = 1
                            anchor_rpn_classes_target[0, ay, ax, anchorShapebox_idx*2 + 1 ] = 0

                            # Regressors 라벨링
                            # GT bounding의 중심좌표,너비,높이 와 현재 앵커 사이의 중심좌표,너비,높이
                            # 차이를 계산(deltas)하여 라벨링
                            # RPN 학습시 예측 앵커 중심좌표와 타깃앵커중심좌표 차이(deltas)가 가장 없도록 학습
                            # ==> 즉, GT bounding box 중심 좌표에 가장 가까운
                            # 앵커가 예측 생성되도록 목표
                            # [0,0,0,0],[0,0,0,1],[0,0,0,2],[0,0,0,3]....[0,0,0,32][0,0,0,33][0,0,0,34],[0,0,0,35]
                            # [0,36,55,0]~~~[0,36,55,35] 까지  총 4*k = 36개 인덱스 증가 하도록 설정
                            dx, dy, dw, dh = self.deltas_for_AnchorRegressorTargetdata(anchor_box_coord_updated,GT_bounding_box)
                            # 계산된 차이값으로 regressors target 라벨링
                            anchor_rpn_regressors_target[0, ay, ax, anchorShapebox_idx * 4] = dx
                            anchor_rpn_regressors_target[0, ay, ax, anchorShapebox_idx * 4 + 1] = dy
                            anchor_rpn_regressors_target[0, ay, ax, anchorShapebox_idx * 4 + 2] = dw
                            anchor_rpn_regressors_target[0, ay, ax, anchorShapebox_idx * 4 + 3] = dh

                        elif iou <= 0.1: # 개체가 있을 확률이 낮은 앵커박스
                            # classed 라벨링 : 개체가 있을 확률이 낮은 앵커는(배경) 0, 1 저장
                            # 배경으로서 사물이 없는 영역임으로 사물 좌표는 필요없음
                            anchor_rpn_classes_target[0, ay, ax, anchorShapebox_idx * 2] = 0
                            anchor_rpn_classes_target[0, ay, ax, anchorShapebox_idx * 2 + 1] = 1


        #print(len(anchor_IOU_high_list))
        #self.Anchor_box_display(anchor_IOU_high_list, self.image)
        return anchor_rpn_classes_target, anchor_rpn_regressors_target

    def deltas_for_AnchorRegressorTargetdata(self, anchor_box, GT_bounding_box):
        # Gx, Gy, Gw, Gh ==> Ground True box 의  중심좌표(Gx,Gy) 넓이(Gw), 높이(Gh)
        # Ax, Ay, Aw, Ah ==> Anchor box의 중심좌표(Ax,Ay) 넓이(Aw) 높이(Ah)
        # delta(차이 계산공식)
        # Tx = (Gx - Ax) / Aw
        # Ty = (Gy - Ay) / Ah
        # Tw = log(Gw / Aw)
        # Th = log(Gh / Ah)

        # Anchor box의 중심좌표(Ax,Ay) 넓이(Aw) 높이(Ah) 계산
        Ax, Ay, Aw, Ah = self.MinXyMaxXy_to_CPointXyWh(anchor_box)
        # Ground True box 의  중심좌표(Gx,Gy) 넓이(Gw), 높이(Gh) 계산
        Gx, Gy, Gw, Gh = self.MinXyMaxXy_to_CPointXyWh(GT_bounding_box)
        Tx = (Gx - Ax) / Aw
        Ty = (Gy - Ay) / Ah
        Tw = np.log(Gw / Aw)
        Th = np.log(Gh / Ah)
        # print('Anchor delta : ',Tx, Ty, Tw, Th)
        return Tx, Ty, Tw, Th

    def decoding_bbox_by_deltas(self,anchor_box, deltas):
        #print('decoding_bbox_by_deltas')
        # print('decode_bbox_by_deltas')
        # anchor_box 와 rpn 예측 deltas 를 이용한 region proposal 앵커 좌표 복원
        ax, ay, aw, ah = self.MinXyMaxXy_to_CPointXyWh(anchor_box)
        Tx, Ty, Tw, Th = deltas

        # deltas(차이값)을 이용해 예측 bounding box의 중심좌표 복원
        Px = Tx * aw + ax
        Py = Ty * ah + ay
        # deltas(차이값)을 이용해 예측 boundig box의 넓이, 높이 복원
        # Pw = (np.exp(Tw)) * aw
        # Ph = (np.exp(Th)) * ah
        Pw = (np.e ** Tw) * aw
        Ph = (np.e ** Th) * ah

        # 예측 bounding box의 중심좌표와 넓이,높이로 xmin,ymin,xmax,ymax 좌표 획득
        xmin, ymin, xmax, ymax = self.CPointXyWh_to_MinXyMaxXy([Px, Py, Pw, Ph])
        return xmin, ymin, xmax, ymax

    def MinXyMaxXy_to_CPointXyWh(self, box):
        # 박스의 (xmin,ymin,xmax,ymax) 좌표를 가지고
        # 박스의 중심좌표(centerpoint)와 넓이, 높이를 계산하여 반환
        xmin, ymin, xmax, ymax = box
        w = xmax - xmin
        h = ymax - ymin
        x = xmin + w/2
        y = ymin + h/2
        return x, y, w, h

    def CPointXyWh_to_MinXyMaxXy(self, a):
        x, y, w, h = a
        xmin = x - w / 2
        ymin = y - h / 2
        #xmax = x + w
        #ymax = y + h
        xmax = w + xmin
        ymax = h + ymin

        return xmin, ymin, xmax, ymax

    def iou_calculation(self, a, b):
        # print('iou_calculation func')
        # iou ==> 두 박스의   겹치는영역(intersection:교집합) / 전체영역(union:합집합)
        # iou ==> 1 일 경우 두 박스가 완전히 겹치는 박스가 되는 것임
        return self.intersection(a, b) / self.union(a, b)

    def intersection(self, a, b):
        # The area intersection(교집합) 계산 of 두(anchorbox, GTbox)경계 박스
        axmin, aymin, axmax, aymax = a  # anchorbox list unpacking
        bxmin, bymin, bxmax, bymax = b  # GTbox list unpacking

        # W(너비), H(높이) 기준 교집합 영역 검출
        xmin = np.maximum(axmin, bxmin)
        ymin = np.maximum(aymin, bymin)
        xmax = np.minimum(axmax, bxmax)
        ymax = np.minimum(aymax, bymax)
        w = xmax - xmin
        h = ymax - ymin
        if w <= 0 or h <= 0:   # 교집합 영역이 없을 경우
            return 0
        if w > 0 and h > 0:   # 교집합 영역이 있을 경우
            return w * h  # 교집합 영역 크기 계산

    def union(self, a, b):
        # The area union(합집합) 계산 of 두(anchorbox, GTbox)경계 박스
        # 합집합 = a박스의 넓이 + b박스의 넓이  - (a,b)박스의 교집합
        # A ∪ B = A + B − (A ∩ B)
        axmin, aymin, axmax, aymax = a
        bxmin, bymin, bxmax, bymax = b
        a_area = np.abs(axmax - axmin) * np.abs(aymax - aymin)
        b_area = np.abs(bxmax - bxmin) * np.abs(bymax - bymin)
        return a_area + b_area - self.intersection(a, b)

    def Create_anchor_Shape_by_ScaleRatio(self):
        # 그냥 base 132 사이즈를 기준으로 형태가 다른 9개의 임의의 앵커 박스 생성
        # 추후 이미지 크기 적용 앵커 박스 형성하도록 수정할 예정!!
        print('Create_9개의 anchor_Shape_by_ScaleRatio!!')
        scales, ratios = np.meshgrid(self.anchor_box_scales, self.anchor_box_ratios)

        scales = np.ravel(scales)
        ratios = np.ravel(ratios)
        ratios = np.sqrt(ratios)

        heights = scales * ratios * self.anchor_box_base_size
        widths = scales / ratios * self.anchor_box_base_size

        ch = heights / 2
        cw = widths / 2
        anchorboxes = np.column_stack((
            0 - cw,  # xmin
            0 - ch,  # ymin
            0 + cw,  # xmax
            0 + ch  # ymax
        ))

        #print(anchorboxes)
        #self.Anchor_box_display(anchorboxes, self.image)
        return anchorboxes

    def Anchor_box_display(self, anchorlist, image):
        for box in anchorlist:
            x_min = int(box[0])
            y_min = int(box[1])
            x_max = int(box[2])
            y_max = int(box[3])
            cv2.rectangle(np.squeeze(image), (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=3)
        plt.imshow(np.squeeze(image))
        plt.show()





