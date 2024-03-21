import os
import numpy as np
import cv2
import time
from .bg import boundary_guided 

def bg_infer(score, textMask, kernelMap, distMap, min_score, min_area, scale=1):
    if scale>1:
        kernelMap = kernelMap[::scale, ::scale]
    result = boundary_guided(score, textMask, kernelMap, distMap,
                              min_score, min_area, scale)
    bboxes = result[0]
    scores = result[1]
    posttime_ms = result[2]
    bboxes_list = []
    scores_list = []
    for box, score in zip(bboxes, scores):
        bboxes_list.append(np.array(box).reshape((-1,2)))
        scores_list.append(np.array(score))
    return bboxes_list, scores_list, posttime_ms 
