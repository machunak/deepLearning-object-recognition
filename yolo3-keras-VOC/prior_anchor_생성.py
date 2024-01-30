import numpy as np
import tensorflow as tf
anchors = np.array([[1.25, 1.625],
                    [2, 3.75],
                    [4.125, 2.875]])
print(anchors)

anchors_xywh = np.zeros((3,4))
print(anchors_xywh)
anchors_xywh[:,2:4] = anchors
print(anchors_xywh)

boxes2 = tf.concat([anchors_xywh[..., :2] - anchors_xywh[..., 2:] * 0.5,
                    anchors_xywh[..., :2] + anchors_xywh[..., 2:] * 0.5], axis=-1)
print(boxes2)

# [[11.4375 32.25    5.375  14.5   ]]