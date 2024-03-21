import math
import time

import cv2
import numpy as np
import torch
import torch.nn as nn

from ..loss import build_loss, iou, ohem_batch
from ..post_processing import expand_poly_ccl, bg_infer

class CBN_Head(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_classes, loss_text,
                 loss_kernel, loss_emb, loss_distance):
        super(CBN_Head, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               hidden_dim,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(hidden_dim,
                               num_classes,
                               kernel_size=1,
                               stride=1,
                               padding=0)

        # Context-aware module (CAM)
        self.conv_dist = nn.Conv2d(hidden_dim, 2, kernel_size=1, stride=1, padding=0)

        self.text_loss = build_loss(loss_text)
        self.kernel_loss = build_loss(loss_kernel)
        self.emb_loss = build_loss(loss_emb)
        self.distance_loss = build_loss(loss_distance)
        
        self.smoothl1loss = torch.nn.SmoothL1Loss()
        self.l1loss = torch.nn.L1Loss()
       
        num_pixels = 8 
        self.conv_fuse = nn.Conv2d(num_pixels*2, num_pixels, kernel_size=1, stride=1, padding=0)
        self.bn_fuse = nn.BatchNorm2d(num_pixels)
        self.relu_fuse = nn.ReLU(inplace=True)

        self.convout1 = nn.Conv2d(num_pixels*2, num_pixels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_pixels)
        self.relu2 = nn.ReLU(inplace=True)
        self.convout2 = nn.Conv2d(num_pixels, 2, kernel_size=1, stride=1, padding=0)
        
        self.conv_pt = nn.Conv2d(hidden_dim, num_pixels, kernel_size=3, stride=1, padding=1)
        self.bn_pt = nn.BatchNorm2d(num_pixels)
        self.relu_pt = nn.ReLU(inplace=True)
 
        self.phi= self.conv1d(8, 4)
        self.psi = self.conv1d(8, 4)
        self.delta = self.conv1d(num_pixels, num_pixels//2)
        self.rho = self.conv1d(num_pixels//2, num_pixels)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def conv1d(self, in_channel, out_channel):
        layers = [
            nn.Conv1d(in_channel, out_channel, 1, bias=False),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(inplace=True),
        ]
        return nn.Sequential(*layers)

    def forward(self, f):
        out = self.conv1(f)
        out = self.relu1(self.bn1(out))

        out_seg = self.conv2(out)

        norm_outseg = torch.sigmoid(out_seg[:, :2, :, :])
        
        # branch 2
        out_dist = self.conv_dist(out)
        dist_branch = out_dist
        
        # branch 3
        out_pixels = self.conv_pt(out)
        out_pixels = self.relu_pt(self.bn_pt(out_pixels))
        
        # GL-CAM part
        batch, n_class, height, width = norm_outseg.shape
        seg_flat = norm_outseg.view(batch, n_class, -1)
        M = seg_flat
        channel = out_pixels.shape[1]
        pixels_flat = out_pixels.view(batch, channel, -1)
        feat_k = (M @ pixels_flat.transpose(1, 2)).transpose(1, 2)   # (batch, 8, 2)

        query = self.phi(feat_k).transpose(1,2)
        key = self.psi(pixels_flat)
        logit = query @ key
        attn_global = torch.softmax(logit, 1)
       
        delta = self.delta(feat_k)   # (batch, 4, 2)
        
        out_dist = torch.sigmoid(out_dist)
        attn_local = out_dist.view(batch, 2, -1)   # (batch, 2, hw)

        attn_sum_global = delta @ attn_global  # (batch, 4, hw)
        attn_sum_local = delta @ attn_local  # (batch, 4, hw)

        seg_obj_global = self.rho(attn_sum_global).view(batch, -1, height, width)  # (batch 8, h, w)
        seg_obj_local = self.rho(attn_sum_local).view(batch, -1, height, width)  # (batch 8, h, w)

        fuse_concat = torch.cat([seg_obj_global, seg_obj_local], 1)   # (batch, 16, h, w)
        fuse_aug = self.conv_fuse(fuse_concat)
        fuse_aug = self.relu_fuse(self.bn_fuse(fuse_aug))

        concat = torch.cat([out_pixels, fuse_aug], 1)   # (batch, 16, h, w)
        seg_aug = self.convout1(concat)
        seg_aug = self.relu2(self.bn2(seg_aug))
        seg_aug = self.convout2(seg_aug)
        
        # return out
        return [out_seg, dist_branch, seg_aug]

    def get_results(self, out, img_meta, cfg):
        outputs = dict()
        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            start = time.time()

        torch.cuda.synchronize()
        start = time.time()
        out_np = out.cpu().numpy()[0].astype(np.float32)
        t_mid = time.time()
        score = out_np[0, :, :]
        kernels = out_np[1:3, :, :].astype(np.uint8)
        text_mask = out_np[1, :, :].astype(np.uint8)
        dist = out_np[-1, :, :].astype(np.float32)

        #----------------------------------------#
        org_img_size = img_meta['org_img_size'][0]
        img_size = img_meta['img_size'][0]
        # if torch.is_tensor(img_size):
        #     img_size = img_size.numpy()

        scale = (float(org_img_size[1]) / float(img_size[1]),
                 float(org_img_size[0]) / float(img_size[0]))

        bboxes = []
        scores = []
        prepare_t = time.time()
        # print('prepare time: {} + {} = {}'.format(t_mid-start, prepare_t-t_mid, prepare_t-start))
        #scaled_bboxes, scaled_scores, post_time = bg_infer(score, text_mask, kernels[-1, :, :], dist,   # cpp version
        #                                               cfg.test_cfg.min_score, cfg.test_cfg.min_area)
        scaled_bboxes, scaled_scores, post_time = expand_poly_ccl(score, text_mask, kernels[-1, :, :], dist.astype(np.float),   # python version
                                                       cfg.test_cfg.min_score, cfg.test_cfg.min_area)
        
        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(det_post_time = post_time))

        for expand_box, score in zip(scaled_bboxes, scaled_scores):
            if cfg.test_cfg.bbox_type == 'rect':
                rect = cv2.minAreaRect(expand_box)
                bounding_box = cv2.boxPoints(rect)
                bounding_box = np.int0(bounding_box)*scale
            elif cfg.test_cfg.bbox_type == 'poly':
                expand_box = expand_box.astype(np.int32)*scale
                bounding_box = expand_box

            bbox = bounding_box.astype('int32')
            bboxes.append(bbox.reshape(-1))
            scores.append(score)
        
        outputs.update(dict(bboxes=bboxes, scores=scores))
        return outputs

    def loss(self, out, gt_texts, gt_kernels, gt_distmap, training_masks, gt_instances,
             gt_bboxes):
        # output
        texts = out[0][:, 0, :, :]
        kernels = out[0][:, 1:2, :, :]
        embs = out[0][:, 2:, :, :]
        distmap = out[1]
        aug_segT = out[2][:, 0, :, :]
        aug_segK = out[2][:, 1:2, :, :]

        # text loss
        selected_masks = ohem_batch(aug_segT, gt_texts, training_masks)
        loss_text = self.text_loss(texts,
                                   gt_texts,
                                   selected_masks,
                                   reduce=False)
        iou_text = iou((texts > 0).long(),
                       gt_texts,
                       training_masks,
                       reduce=False)

        loss_augtext = self.text_loss(aug_segT, gt_texts, selected_masks, reduce=False)
        iou_augtext = iou((aug_segT > 0).long(), gt_texts, training_masks, reduce=False)

        losses = dict(loss_text=loss_text, iou_text=iou_text,
            loss_augtext=loss_augtext,
            iou_augtext=iou_augtext)

        # dist loss
        loss_dists = []
        for i in range(distmap.size(1)):
            distmap_i = distmap[:, i, :, :]
            gt_distmap_i = gt_distmap[:, i, :, :]
            loss_dist = self.distance_loss(distmap_i, gt_distmap_i, gt_texts*training_masks)
            loss_dists.append(loss_dist.view(-1))
        loss_dist = torch.mean(torch.stack(loss_dists, dim=1), dim=1)
        losses.update(dict(loss_dist=loss_dist))

        # kernel loss
        loss_kernels = []
        loss_augkernels = []
        selected_masks = gt_texts * training_masks
        for i in range(kernels.size(1)):
            kernel_i = kernels[:, i, :, :]
            gt_kernel_i = gt_kernels[:, i, :, :]
            loss_kernel_i = self.kernel_loss(kernel_i,
                                             gt_kernel_i,
                                             selected_masks,
                                             reduce=False)
            loss_kernels.append(loss_kernel_i)

            augkernel_i = aug_segK[:, i, :, :]
            loss_augkernel_i = self.kernel_loss(augkernel_i, gt_kernel_i, selected_masks, reduce=False)
            loss_augkernels.append(loss_augkernel_i)            

        loss_kernels = torch.mean(torch.stack(loss_kernels, dim=1), dim=1)
        loss_augkernels = torch.mean(torch.stack(loss_augkernels, dim=1), dim=1)
        iou_kernel = iou((kernels[:, -1, :, :] > 0).long(),
                         gt_kernels[:, -1, :, :],
                         training_masks * gt_texts,
                         reduce=False)
        
        iou_augkernel = iou((aug_segK[:, -1, :, :] > 0).long(), gt_kernels[:, -1, :, :], training_masks * gt_texts, reduce=False)

        losses.update(dict(loss_kernels=loss_kernels, iou_kernel=iou_kernel,
             loss_augkernels=loss_augkernels,
            iou_augkernel=iou_augkernel,))

        # embedding loss
        loss_emb = self.emb_loss(embs,
                                 gt_instances,
                                 gt_kernels[:, -1, :, :],
                                 training_masks,
                                 gt_bboxes,
                                 reduce=False)
        losses.update(dict(loss_emb=loss_emb))

        return losses
