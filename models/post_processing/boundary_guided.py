import cv2
import numpy as np
import pdb
import math
import time
import Polygon as plg
import pyclipper
from numpy.linalg import norm
 
def getroi(contour, height, width, appenddix=10):
    x, y, w, h = cv2.boundingRect(contour)
    ######----------------------------######
    leftx = x 
    bottomx = x + w 
    lefty = y
    bottomy = y + h

    newlx = leftx - appenddix
    newly = lefty - appenddix
    newbx = bottomx + appenddix 
    newby = bottomy + appenddix

    if newlx < 0:
        newlx = 0
    if newly < 0:
        newly = 0
    if newbx > width-1:
        newbx = width-1 
    if newby > height-1:
        newby = height-1
    return [newlx, newly, newbx, newby] 
   
def unclip_dist(box, distance):
    box = box.astype(np.int32)
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = offset.Execute(distance)
    return expanded

def expand_poly_ccl(score, text_mask, kernelmap, distmap, min_score, min_area, scale=1):
    funstart = time.time()
    expand_bbox = []
    bbox_score = []
    height, width = score.shape
    tstart = time.time()
    if scale>1:
        kernelmap = kernelmap[::scale, ::scale]
    # label_num, label = cv2.connectedComponents(kernelmap, connectivity=4)
    label_num, label, stats, centroids = cv2.connectedComponentsWithStats(kernelmap, connectivity=4)
    t0 = time.time() - tstart
    # print('ccl time: {}'.format(time.time() - tstart))
    iter_time = 0.0
    for i in range(1, label_num):
        tstart = time.time()
        iterstart = time.time()
        x, y, width, height, _ = stats[i]
        ker_roi = label[y:y+height, x:x+width]
        indkernel = ker_roi == i    # need roi
        contours, _ = cv2.findContours((indkernel*255).astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE, offset=(x, y))
       
        if len(contours) == 0:
            # print('roi no contours filter')
            continue
        cnt = contours[0].reshape((-1,2))
        oricnt = cnt
        if scale>1:
            cnt = np.dot(scale, cnt)
        avliable_dist = distmap[cnt[:, 1], cnt[:, 0]]
        coords = np.array(np.where(avliable_dist>0)).transpose((1,0))
        if coords.shape[0] == 0:
            continue
            
        offsetdist = np.mean(avliable_dist[coords[:,0]])
        tstart = time.time()
        expandpoly = unclip_dist(cnt.reshape((-1,2)), offsetdist)
        if len(expandpoly) == 0:
            continue
        # print('expand poly num: {}'.format(len(expandpoly)) )
        expand_box = np.array(expandpoly[0]).reshape((-1, 1, 2))
        newroi = getroi(expand_box, label.shape[0]*scale, label.shape[1]*scale)    
        text_line = np.zeros((newroi[3]-newroi[1], newroi[2]-newroi[0]), dtype='uint8')
        roi_mask = text_mask[newroi[1]:newroi[3], newroi[0]:newroi[2]]
        roi_score = score[newroi[1]:newroi[3], newroi[0]:newroi[2]]
        cv2.drawContours(text_line, [expand_box], -1, i, -1, offset=(-newroi[0], -newroi[1]))
        text_line = text_line * roi_mask
        ind = text_line == i
        # print('after get_dist and draw and mask time: {}'.format(time.time() - tstart))
        
        tstart = time.time()
        points = np.array(np.where(ind)).transpose((1, 0))
        
        if points.shape[0] < min_area:
            # print('min_area filter: {}'.format(points.shape[0]))
            continue

        score_i = np.mean(roi_score[ind])
        if score_i < min_score:
            # print('min_score filter: {}'.format(score_i))
            continue
        iter_time += (time.time()-iterstart)
        res_contour, _ = cv2.findContours((ind*255).astype(np.uint8), cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE, offset=(newroi[0], newroi[1]))
        
        # total-text img1098 case need this
        max_nums = 0
        max_index = 0
        for kk, xcnt in enumerate(res_contour):
            if xcnt.shape[0] > max_nums:
                max_nums = xcnt.shape[0]
                max_index = kk
    
        rcnt = res_contour[max_index].reshape((-1,2))
        # print('final part time: {}'.format(time.time() - tstart))        
        expand_bbox.append(rcnt)
        bbox_score.append(score_i)

    # print('expand_poly_ccl time: {}'.format(time.time() - funstart))
    # print('return post time: {}'.format(t0+iter_time))
    return expand_bbox, bbox_score, t0+iter_time


