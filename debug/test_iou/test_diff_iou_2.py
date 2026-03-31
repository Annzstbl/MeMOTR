import copy

import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
from mmcv.ops import diff_iou_rotated_2d
import os

'''
When the w deviation of the prediction box changes within a certain range, 
the IoU calculated using 'rbbox_overlaps' and 'diff_iou_rotated_2d' jumps instead of continuous, 
which is very different from the results obtained using the intersection and union ratio method of pixels.
'''

original_box = (339.15, 230.95, 308.218364151133, 33.12346705650146, 1.5331517813945545)
xc, yc, w, h, ag = original_box

RIoU_list = []
RIoU2_list = []
IoU_list = []
w_deviation_list = range(-20, 21)

for w_deviation in w_deviation_list:
    prediction_box = (xc, yc, w + w_deviation, h, ag)

    # Caculate IoU using rbbox_overlaps
    original_box_tensor = torch.Tensor(original_box).unsqueeze(0)
    prediction_box_tensor = torch.Tensor(prediction_box).unsqueeze(0)

    # Caculate IoU using diff_iou_rotated_2d
    device = torch.device('cuda:0')
    original_box_tensor_GPU = original_box_tensor.unsqueeze(0).to(device)
    prediction_box_tensor_GPU = prediction_box_tensor.unsqueeze(0).to(device)
    RIoU2 = diff_iou_rotated_2d(original_box_tensor_GPU, prediction_box_tensor_GPU)
    RIoU2_list.append(RIoU2.item())

    # Caculate IoU using the intersection and union ratio method of pixels
    # x, y, w, h, ag -> p1, p2, p3, p4
    xc2, yc2, w2, h2, ag2 = prediction_box
    wx2, wy2 = w2 / 2 * math.cos(ag2), w2 / 2 * math.sin(ag2)
    hx2, hy2 = -h2 / 2 * math.sin(ag2), h2 / 2 * math.cos(ag2)
    p1_new = (xc2 + wx2 - hx2, yc2 - wy2 + hy2)
    p2_new = (xc2 - wx2 - hx2, yc2 + wy2 + hy2)
    p3_new = (xc2 - wx2 + hx2, yc2 + wy2 - hy2)
    p4_new = (xc2 + wx2 + hx2, yc2 - wy2 - hy2)
    ps_new = [p1_new, p2_new, p3_new, p4_new]

    original_grasp_bboxes = np.array([[[328.4, 76.4], [361.5, 77.6], [349.9, 385.6], [316.8, 384.3]]], dtype=np.int32)      # 4-point representation of original box
    prediction_grasp_bboxes = np.array([ps_new], dtype=np.int32)        # 4-point representation of prediction box
    im = np.zeros((512, 640), dtype="uint8")
    im1 = np.zeros((512, 640), dtype="uint8")
    original_grasp_mask = cv2.fillPoly(im, original_grasp_bboxes, 255)          # original box
    prediction_grasp_mask = cv2.fillPoly(im1, prediction_grasp_bboxes, 255)     # prediction box
    masked_and = cv2.bitwise_and(original_grasp_mask, prediction_grasp_mask, mask=im)   # intersection
    masked_or = cv2.bitwise_or(original_grasp_mask, prediction_grasp_mask)      # union

    or_area = np.sum(np.float32(np.greater(masked_or, 0)))
    and_area = np.sum(np.float32(np.greater(masked_and, 0)))
    IOU = and_area / or_area
    IoU_list.append(IOU)


plt.figure()
plt.plot(w_deviation_list, RIoU2_list, label='Caculate results in diff_iou_rotated_2d')
plt.plot(w_deviation_list, IoU_list, label='Caculate results in intersection and union ratio of pixels')
plt.legend()

save_file_path = '/data/users/wangying01/lth/hsmot/hsmot/test/test_diff_iou/diff_iou_2.png'
os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
plt.savefig(save_file_path)
plt.close()