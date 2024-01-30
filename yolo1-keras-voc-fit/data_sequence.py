import math
import cv2 as cv
import numpy as np
import os
from tensorflow import keras
import matplotlib.pyplot as plt

class SequenceData(keras.utils.Sequence):

    #def __init__(self, model, dir, target_size, batch_size, shuffle=True):
    def __init__(self, model, dir, target_size, batch_size):
        self.model = model
        self.datasets = []
        if self.model == 'train':
            with open(os.path.join(dir, 'train_2007.txt'), 'r') as f:
                self.datasets = self.datasets + f.readlines()
                #print("len train self.datasets : ", len(self.datasets))  # 2501
        elif self.model == 'val':
            with open(os.path.join(dir, 'val_2007.txt'), 'r') as f:
                self.datasets = self.datasets + f.readlines()
                #print("len val self.datasets : ", len(self.datasets))  # 2510
        self.image_size = target_size[0:2]
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.datasets))
        #print('indexes : ', self.indexes)
        #self.shuffle = shuffle  # 1 epoch 이후 index shuffle 설정

    def __len__(self):
        # num_imgs / batch_size 크기 ==> step size 만큼 __getitem__( ) 호출
        # 결국, num_imgs / batch_size 값은 ==> 1 epoch의 시퀀스 step size
        num_imgs = len(self.datasets)
        return math.ceil(num_imgs / float(self.batch_size))  # step size

    def __getitem__(self, idx):
        # __len__() 리턴값인 step size회수 만큼 반복 호출
        # fit()의 shuffle=True 일 경우 __getitem__(self, index)에 전달되는
        # index는  __len__()의 반환값인 step size -1 값 중 랜덤 index를 전달
        # fit()의 shuffle=False 일 경우 1 epoches 당 0 부터  __len()__의 리턴값인 step size - 1 까지
        # 순차적인 index를 __getitem__() 로 전달

        # 전달된 idx를 활용해 추출할 batch_indexs 생성
        #print('idx : ', idx)
        batch_indexs = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]

        # 생성된 batch_indexs 인덱스 활용 datasets 추출
        batch = [self.datasets[k] for k in batch_indexs]

        # batch 크기만큼 x(이미지), y(label) 생성
        X, y = self.data_generation(batch)
        #print('==============per batch ===============')
        return X, y

    # def on_epoch_end(self):  # 1 epoch 끝 동작 마다 호출되는 함수
    #     # 한 epoch 이후 인덱스 서플
    #     if self.shuffle:
    #         np.random.shuffle(self.indexes)

    def new_get_real_bounding_boxes(self,labellist,w_chanage_ratio,h_chanage_ratio ):
        new_label_list = []
        for lb in labellist:
            l = lb.split(',')
            l = np.array(l, dtype=np.int)
            l[0] = l[0] * w_chanage_ratio
            l[1] = l[1] * h_chanage_ratio
            l[2] = l[2] * w_chanage_ratio
            l[3] = l[3] * h_chanage_ratio
            new_label_list.append(l)
        return new_label_list

    def read(self, dataset):
        dataset = dataset.strip().split()
        image_path = dataset[0]
        label = dataset[1:]
        #print('src label : ', label)
        image = cv.imread(image_path)
        # RGB원본 이미지를 opencv에서 imread()함수이용한 이미지 로딩 배열은 BGR 형태
        # BGR을 원래 RGB형태로 변환해야 원본 이미지 색생이 나옴
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image_h, image_w = image.shape[0:2]
        #print('src image_h, src image_w : ', image_h, image_w)
        image = cv.resize(image, self.image_size)
        image = image / 255.
        # 이미지 resize 따른 target label 좌표도 변경됨에 따라 GT label 데이터 조정
        h_chanage_ratio = image.shape[0]/image_h
        w_chanage_ratio = image.shape[1]/image_w

        label = self.new_get_real_bounding_boxes(label,w_chanage_ratio,h_chanage_ratio)
        #print('real label : ', label)
        # for lb in label:
        #     cv.rectangle(image, (lb[0],lb[1]), (lb[2], lb[3]), color=(0, 255, 0), thickness=3)
        # plt.imshow(image)
        # plt.show()
        image_resize_h = image.shape[0]
        image_resize_w = image.shape[1]
        #print('coord image_h, coord image_w : ', image_resize_h, image_resize_w)
        label_matrix = np.zeros([7, 7, 25])
        for l in label:
            #print('l : ', l)
            xmin = l[0]
            ymin = l[1]
            xmax = l[2]
            ymax = l[3]
            cls = l[4]
            x = (xmin + xmax) / 2 / image_resize_w  # 448 크기로 나눈 중심좌표 x
            y = (ymin + ymax) / 2 / image_resize_h  # 448 크기로 나눈 중심좌표 y
            w = (xmax - xmin) / image_resize_w # 448 크기로 나눈 bbox 너비
            h = (ymax - ymin) / image_resize_h  # 448 크기로 나눈 bbox 높이
            #print('x , y, w , h : ', x, y, w, h)
            loc = [7 * x, 7 * y]  # x, y 중심 좌표 점
            loc_i = int(loc[1])  # 행 이동 좌표
            loc_j = int(loc[0])  # 열 이동 좌표
           # print('loc_i , loc_j : ', loc_i, loc_j)
            y = loc[1] - loc_i   # 한 grid cell 기준으로 중심좌표 encoding
            x = loc[0] - loc_j   # 한 grid cell 기준으로 중심좌표 encoding
           # print("coord x, y , w, h: ", x, y, w, h)
            if label_matrix[loc_i, loc_j, 4] == 0:
                label_matrix[loc_i, loc_j, cls + 5] = 1  # 인덱스 5를 기준으로 해당 클래스 위치만 1로 설정
                label_matrix[loc_i, loc_j, :4] = [x, y, w, h]
                label_matrix[loc_i, loc_j, 4] = 1  # loss function ==> response mask

        return image, label_matrix



    def data_generation(self, batch_datasets):
        images = []
        labels = []

        for dataset in batch_datasets:  # batch dataset 만큼 이미지,라벨 로딩
            image, label = self.read(dataset)
            images.append(image)
            labels.append(label)

        X = np.array(images)
        y = np.array(labels)

        return X, y
