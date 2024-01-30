import numpy as np
import cv2
import matplotlib.pyplot as plt

class MenagementCls():
    def start_imgload_training(self, trainObj, training_dataset_folder):
        print('start_imgload_training!!')
        # numpy로 저장된 이미지 파일 이름 정보 로딩
        img_filename = np.load(training_dataset_folder+'cars_train.npy')
        #print(len(img_filename)) # 8144
        #print(img_filename[0])
        #img = cv2.imread(training_dataset_folder+'cars_train/'+img_filename[0])

        # 이미지 데이터 annotation 정보 로딩
        img_annotation_dict = np.load(training_dataset_folder+'cars_train_annos.npy', allow_pickle=True).item()
        # print(img_annotation_dict) # [min_x, min_y, max_x, max_y, id]
        # print(img_annotation_dict['00001.jpg']) #[[39, 116, 569, 375, 14]]
        # # #cv2.rectangle(사각형을 넣을 이미지, 사각형좌측상단좌표,사각형우측하단좌표,테두리선 색상,테두리선 두께)
        # cv2.rectangle(img, (39, 116), (569, 375), color=(0, 255, 0), thickness=3)
        # plt.imshow(img) # figure창에 이미지 랜더링
        # plt.show() # 랜더링 이미지 화면 출력
        img_cnt = 0
        while(img_cnt < 50):
            imgName = img_filename[img_cnt]
            img = cv2.imread(training_dataset_folder+'cars_train/'+imgName)
            img_bounding_boxes = []  # bounding box 정보 저장 list
            for annot in img_annotation_dict[imgName]:
                xmin, ymin, xmax, ymax = annot[0:4] # list unpacking
                img_bounding_boxes.append([xmin, ymin, xmax, ymax])
            img_class_names = ['car'] # Detector 클래스 구분 이름
            print('원본 이미지 shape : ', img.shape)  # 첫번째 원본 이미지 Shape : (400, 600, 3)

            # 사이즈가 다른 이미지들의 크기 갱신 및 바운딩 박스 정보 갱신
            trainObj.Update_image(img, img_bounding_boxes, img_class_names)

            img_cnt += 1
            print("="*80)

    def start_imgload_testing(self, testingObj, testing_dataset_folder):
        print('start_imgload_testing!!')
        # numpy로 저장된 test 이미지 파일 이름 정보 로딩
        img_filename = np.load(testing_dataset_folder+'cars_test.npy')
        #print(img_filename) # 8144 개
        # ['00001.jpg' '00002.jpg' '00003.jpg' '00004.jpg' '00005.jpg' '00006.jpg'....
        #print(img_filename[0])
        #img = cv2.imread(training_dataset_folder+'cars_train/'+img_filename[0])

        # 이미지 데이터 annotation 정보 로딩
        #img_annotation_dict = np.load(testing_dataset_folder+'cars_test_annos.npy', allow_pickle=True).item()
        # print(img_annotation_dict) # [min_x, min_y, max_x, max_y, id]
        # print(img_annotation_dict['00001.jpg']) #[[39, 116, 569, 375, 14]]
        # # #cv2.rectangle(사각형을 넣을 이미지, 사각형좌측상단좌표,사각형우측하단좌표,테두리선 색상,테두리선 두께)
        # cv2.rectangle(img, (39, 116), (569, 375), color=(0, 255, 0), thickness=3)
        # plt.imshow(img) # figure창에 이미지 랜더링
        # plt.show() # 랜더링 이미지 화면 출력
        img_cnt = 0
        while(img_cnt < 3):
            imgName = img_filename[img_cnt]
            img = cv2.imread(testing_dataset_folder+'cars_test/'+imgName)
            img_class_names = ['car'] # Detector 클래스 구분 이름
            #print('원본 이미지 shape : ', img.shape)  # 첫번째 원본 이미지 Shape : (400, 600, 3)

            # 사이즈가 다른 테스트 이미지들의 크기 갱신
            testingObj.test_Update_image(img, img_class_names)

            img_cnt += 1
            print("=============== testing processing ============")