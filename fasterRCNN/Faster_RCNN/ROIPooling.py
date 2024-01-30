import numpy as np

class ROIPoolingCls():
    def __init__(self, mode = 'tf', pool_size=(7,7)):
        # tf : (height , width, channels)
        # th : (channels, height, width)
        print('roi pooling cls init!!')
        self.mode = mode
        self.pool_size = pool_size

    def get_region(self, feature_map, roi_one_boxes): # 특징맵 위 roi 영역 획득
        #print('get region ', feature_map.shape) # (1, 37, 56, 512)
        xmin, ymin, xmax, ymax = roi_one_boxes
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)

        if self.mode == 'tf':
            # (height, width, channels)
            region_slicing = np.squeeze(feature_map)[ymin:ymax, xmin:xmax, :]
            #print('region slicing shape : ', region_slicing.shape)
        elif self.mode == 'th':
            region_slicing = np.squeeze(feature_map)[:, ymin:ymax, xmin:xmax]

        return region_slicing

    def Max_pooling(self, region):
        # tf mode : (height, width, channels) pooling 작업
        # th mode : (channels, height, width) pooling 작업
        pool_height, pool_width = self.pool_size  # (7, 7)
        if self.mode == 'tf':
            region_height, region_width, region_channels = region.shape
            # print(region_height, region_width, region_channels)
            pool = np.zeros((pool_height, pool_width, region_channels))
        elif self.mode == 'th':
            region_channels, region_height, region_width = region.shape
            pool = np.zeros((region_channels, pool_height, pool_width))

        h_step = region_height / pool_height
        w_step = region_width / pool_width

        for h in range(pool_height):
            for w in range(pool_width):

                xmin = w * w_step
                xmax = (w+1) * w_step
                ymin = h * h_step
                ymax = (h+1) * h_step

                xmin = int(xmin)
                xmax = int(xmax)
                ymin = int(ymin)
                ymax = int(ymax)

                if xmin == xmax or ymin == ymax:
                    continue
                if self.mode == 'tf':
                    # (행,열) 축 기준 최대값 찾아서 pool[] 배열에 저장
                    pool[h, w, :] = np.max(region[ymin:ymax, xmin:xmax, :], axis = (0,1))
                elif self.mode == 'th':
                    pool[:, h, w] = np.max(region[:, ymin:ymax, xmin:xmax], axis = (1,2))

        return pool


    def get_Pooled_roi_batch(self, feature_map, roi_boxes_batch):
        # pooling 작업 완료된 roi boxes batch 반환
        # feature_map 위에 256개 roi boxes 정보를 올려 maxpooling 작업 수행
        # 반환 shape :  (256, 7, 7, 512 )
        poollist = []

        for roi_one_boxes in roi_boxes_batch:
            # 특징맵 위 roi 영역 획득
            region = self.get_region(feature_map, roi_one_boxes)
            # 획득 영역 Max pooling 적용
            p = self.Max_pooling(region)
            poollist.append(p)

        return np.array(poollist)
