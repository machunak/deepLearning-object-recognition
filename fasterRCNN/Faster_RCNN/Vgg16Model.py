from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
import numpy as np

# VGG16 모델 중 17 layer 까지 만 잘라서 활용
class Vgg16Cls():
    def __init__(self):
        self.vgg16 = VGG16(include_top=False,
                           weights='imagenet',
                           input_tensor=None,
                           input_shape=None,
                           pooling='max')
    def get_summery(self):
        self.vgg16.summary()

    def get_feature_map(self, image):
        print('get feature map')
        print('vgg16 input shape : ', image.shape) # (1, 600, 900, 3)

        # keras Functional API로  vgg16 layer 잘라내어 모델 설계
        #print(self.vgg16.input)  # Vgg16 input_1 layer 입력 Tensor
        #print(self.vgg16.get_layer('block5_conv3').output)  # Vgg16 block5_conv3 layer 출력 Tensor
        vgg16_croped_model = Model(inputs=self.vgg16.input,
                                   outputs=self.vgg16.get_layer('block5_conv3').output)
        #vgg16_croped_model.summary()
        feature_map = np.array(vgg16_croped_model(image))
        print('vgg16 feature_map output shape : ', feature_map.shape)
        feature_map_shape = feature_map.shape
        return feature_map, feature_map.shape

    def get_featuremap_ChannelNum(self, layer = 17):
        return self.vgg16.layers[layer].output.shape[-1]
