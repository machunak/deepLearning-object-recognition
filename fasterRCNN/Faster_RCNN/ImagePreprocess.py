import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def new_get_real_bounding_boxes(image_height_change_ratio, image_width_change_ratio,bounding_boxes):
    print('new get real bounding boxes!!')
    bounding_boxes = np.array(bounding_boxes)
    print(bounding_boxes.shape)  # (1,4)
    #print(bounding_boxes[:,0]) # xmin
    bounding_boxes = np.column_stack( (bounding_boxes[:,0] * image_width_change_ratio,
                                       bounding_boxes[:,1] * image_height_change_ratio,
                                       bounding_boxes[:,2] * image_width_change_ratio,
                                       bounding_boxes[:,3] * image_height_change_ratio) )

    return bounding_boxes

def reshape_for_CNN(array):
    print('reshape_for_CNN')
    # CNN을 위한 shape 조정
    h, w, c = array.shape
    return np.reshape(array, [1,h,w,c])

def resize_image_side(image, small_side = 600):
    print('resize image')
    height, width, channels = image.shape
    if height < width:
        resize_scale = small_side / height # 이미지 사이즈조절 스케일
        # resize(img, dsize=(width, height) : (가로,세로) )
        # height(세로) 크기를 600으로 고정하고 비율 스케일 만큼 가로 크기 조절
        return cv2.resize(image, dsize=(int(width*resize_scale), small_side))
    else:
        resize_scale = small_side / width  # 이미지 사이즈조절 스케일
        return cv2.resize(image, dsize=(small_side, int(height * resize_scale)))


def prepare_image_bboxes(image, bounding_boxes):
    print('prepare image bboxes!!')

    height = image.shape[-3]  # 세로 400
    width = image.shape[-2]   # 가로 600
    # height 와 width 중 작은면의 이미지가 600이 되도록 이미지 사이즈 조정 함수
    image = resize_image_side(image, small_side = 600)
    print('resize image shape : ', image.shape) # (600, 900, 3)

    # CNN을 위한 shape 조정
    # [1, height, width, channels] shape으로 변경
    image = reshape_for_CNN(image)  # [1,h,w,c]
    print('reshape image :', image.shape)

    # height, width 이미지 크기 조정에 따른 실제 바운딩박스 좌표 정보 업데이트
    new_height = image.shape[1]
    new_width = image.shape[2]
    print('source bounding_boxes : ', bounding_boxes)
    # 수정 height / 원본 height ==> 세로 수정 비율,  수정 width / 원본 width ==>  가로 수정비율
    new_bounding_boxes = new_get_real_bounding_boxes(new_height/height, new_width/width,bounding_boxes)

    return image, new_height, new_width, new_bounding_boxes

def test_prepare_image_bboxes(image):
    print('test prepare image bboxes!!')

    # height = image.shape[-3]  # 세로 400
    # width = image.shape[-2]   # 가로 600
    # height 와 width 중 작은면의 이미지가 600이 되도록 이미지 사이즈 조정 함수
    image = resize_image_side(image, small_side = 600)
    print('resize image shape : ', image.shape) # (600, 900, 3)

    # CNN을 위한 shape 조정
    # [1, height, width, channels] shape으로 변경
    image = reshape_for_CNN(image)  # [1,h,w,c]
    print('reshape image :', image.shape)

    # height, width 이미지 크기 조정에 따른 실제 바운딩박스 좌표 정보 업데이트
    new_height = image.shape[1]
    new_width = image.shape[2]
    #print('source bounding_boxes : ', bounding_boxes)
    # 원본 height / 수정 height ==> 세로 수정 비율,  원본 width / 수정 width ==>  가로 수정비율
    #new_bounding_boxes = new_get_real_bounding_boxes(new_height/height, new_width/width,bounding_boxes)

    return image, new_height, new_width

def new_image_GT_bboxes_display(image, bboxes):
    for box in bboxes:
        x_min = int(box[0])
        y_min = int(box[1])
        x_max = int(box[2])
        y_max = int(box[3])
        # np.squeeze(image) : 길이 1의 축(차원)을 제거
        # (1, 600, 900, 3) ==> ( 600, 900, 3)
        cv2.rectangle(np.squeeze(image),(x_min,y_min),(x_max,y_max),color=(0,255,0),thickness=3)
    plt.imshow(np.squeeze(image))
    plt.show()


def image_anchor_point_display(new_image, new_image_height, new_image_width):
    for anchor_y in range(0, new_image_height - 16, 16):
        for anchor_x in range(0, new_image_width - 16, 16):
            #print(anchor_x, anchor_y)
            cv2.rectangle(np.squeeze(new_image), (anchor_x - 2, anchor_y - 2), (anchor_x + 2, anchor_y + 2),
                          color=(255, 0, 0), thickness=2)
    plt.imshow(np.squeeze(new_image))
    plt.show()

def l1smooth():
    """
    in case of using l1 smooth for rpn and rcnn model
    :return:
    """
    return tf.keras.losses.huber
