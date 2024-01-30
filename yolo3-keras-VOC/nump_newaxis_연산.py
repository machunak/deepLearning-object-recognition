import numpy as np

bbox_xywh = np.array([91.5, 258, 43, 116])
print(bbox_xywh[np.newaxis, :])  # (1,4)
strides = np.array([8,16,32])
print(strides[:,np.newaxis]) # (3,1)

bbox_xywh_scaled = bbox_xywh[np.newaxis, :] / strides[:,np.newaxis]
print(bbox_xywh_scaled)

print(bbox_xywh_scaled[0][np.newaxis,:])
