from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np

class RCNNCls():
    def __init__(self, numberofclasses, input_shape=(7,7,512)):
        self.rcnn_numberofclasses = numberofclasses # 1
        self.rcnn_input_shape = input_shape # (7,7,512)
    def create_model(self):
        """
        create rcnn model.
        4096 neurons 대신 1024 뉴런의 2개 dense 층 사용
        multi-label 분류로 sigmoid +  BinaryCrossentropy 사용
        tf.compat.v1.losses.huber_loss 는 L1 smooth loss 와 유사
        """
        input = Input(shape=self.rcnn_input_shape) # (7, 7, 512)

        out = Flatten()(input)
        out = Dense(1024, activation='relu', name='rcnn_dense1')(out)
        out = Dropout(0.5)(out)
        out = Dense(1024, activation='relu', name='rcnn_dense2')(out)
        out = Dropout(0.5)(out)

        rcnn_classifier = Dense(self.rcnn_numberofclasses + 1, activation='sigmoid', name='rcnn_classifier')(out)
        #rcnn_classifier = Dense(self.rcnn_numberofclasses + 1, activation='softmax', name='rcnn_classifier')(out)

        rcnn_regressor = Dense(4 * (self.rcnn_numberofclasses), activation='linear', name='rcnn_regressor')(out)

        rcnn_model = Model(inputs=input, outputs=[rcnn_classifier, rcnn_regressor])
        #rcnn_model.compile(optimizer='adam', loss=['categorical_crossentropy','logcosh'])
        rcnn_model.compile(optimizer='adam', loss=[tf.keras.losses.BinaryCrossentropy(), tf.compat.v1.losses.huber_loss])
        #rcnn_model.compile(optimizer='adam',
        #                    loss=[tf.keras.losses.CategoricalCrossentropy(), tf.compat.v1.losses.huber_loss])
        #rcnn_model.compile(optimizer='adam', loss=[tf.keras.losses.BinaryCrossentropy(), tf.keras.losses.Huber()]) # warnnig 있음
        #rcnn_model.summary()
        return rcnn_model

    def create_rcnn_detector_target(self,anchors, roi_boxes_batch, bounding_boxes,
                                    image_class_names, classes, iou_threshold = 0.5):
        # image_class_names ==> ['car']
        # classes ==> {'car':1}  ==> 원핫인코딩위한 구분자
        print('roi_boxes_batch shape : ', roi_boxes_batch.shape) # (256, 4 )
        batch_size = roi_boxes_batch.shape[0] # 256
        # roi_boxes_batch 사이즈 만큼의 개체 개수 + 배경  ==> numberofclasses + 1 ==> rcnn_classes 구성
        # roi_boxes_batch 사이즈 만큼의 개체 개수 별 4개의 reg boxes ==>  4 * numberofclasses ==> rcnn_regression 구성
        detector_classes_target = np.zeros((batch_size, self.rcnn_numberofclasses+1)) # (256,2)
        detector_regression_target = np.zeros((batch_size, 4*self.rcnn_numberofclasses)) # (256,4)

        #print(detector_classes_target.shape, detector_regression_target.shape)
        for roi_idx in range(batch_size):
            roi = roi_boxes_batch[roi_idx]

            for bbox_idx in range(bounding_boxes.shape[0]):
                bounding_box = bounding_boxes[bbox_idx]
                # roi 와 GT bounding_box 의 iou계산
                iou = anchors.iou_calculation(roi, bounding_box)

                if iou > iou_threshold:  # 0.5
                    class_index = classes[image_class_names[bbox_idx]] # 'car' : 1
                    # class 타깃 원핫인코딩 변환 저장 : [0 1]
                    detector_classes_target[roi_idx] = tf.keras.utils.to_categorical(class_index, self.rcnn_numberofclasses+1)
                    # roi 와 GT bboxes 간의 reg deltas 계산후 타깃 저장
                    dx, dy, dw, dh = anchors.deltas_for_AnchorRegressorTargetdata(roi, bounding_box)
                    #print(dx, dy, dw, dh) # 0.006666666666666667 0.11375 -0.12405264866997888 -0.4346364854084443
                    detector_regression_target[roi_idx, 4 * (class_index-1)] = dx
                    detector_regression_target[roi_idx, 4 * (class_index - 1)+1] = dy
                    detector_regression_target[roi_idx, 4 * (class_index - 1)+2] = dw
                    detector_regression_target[roi_idx, 4 * (class_index - 1)+3] = dh
                else:
                    bg = np.zeros((self.rcnn_numberofclasses+1)) # [ 0 0 ]
                    bg[0] = 1 # [1 0]
                    detector_classes_target[batch_size-1] = bg

        return detector_classes_target, detector_regression_target




