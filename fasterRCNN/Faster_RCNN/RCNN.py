from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np

class RCNNCls():
    def __init__(self, numberofclasses, input_shape=(7,7,512)):
        print('RCNNCls _init_!!')
        self.rcnn_numberofclasses = numberofclasses
        self.rcnn_input_shape = input_shape
    def create_model(self):
        """
        create rcnn model. since the output is way smaller than the way that the original paper uses rcnn. we use
        two dense layers with 1024 neurons instead of 4096 neurons. logcosh is very similar to l1 smooth as the paper
        suggests using
        :return:
        """
        input = Input(shape=self.rcnn_input_shape) # (7, 7, 512)

        out = Flatten()(input)
        out = Dense(1024, activation='relu', name='rcnn_dense1')(out)
        out = Dropout(0.5)(out)
        out = Dense(1024, activation='relu', name='rcnn_dense2')(out)
        out = Dropout(0.5)(out)

        rcnn_classifier = Dense(self.rcnn_numberofclasses + 1, activation='sigmoid', name='rcnn_classifier')(out)
        #rcnn_classifier = Dense(self.RCnn_numberofclasses + 1, activation='softmax', name='rcnn_classifier')(out)

        rcnn_regressor = Dense(4 * (self.rcnn_numberofclasses), activation='linear', name='rcnn_regressor')(out)

        rcnn_model = Model(inputs=input, outputs=[rcnn_classifier, rcnn_regressor])
        #rcnn_model.compile(optimizer='adam', loss=['categorical_crossentropy','logcosh'])
        rcnn_model.compile(optimizer='adam', loss=[tf.keras.losses.BinaryCrossentropy(), tf.compat.v1.losses.huber_loss])
        # rcnn_model.compile(optimizer='adam',
        #                    loss=[tf.keras.losses.CategoricalCrossentropy(), tf.compat.v1.losses.huber_loss])
        #rcnn_model.compile(optimizer='adam', loss=[tf.keras.losses.BinaryCrossentropy(), tf.keras.losses.Huber()]) # warnnig 있음
        #rcnn_model.summary()
        return rcnn_model
