import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone
from .head import build_head
from .neck import build_neck
from .utils import Conv_BN_ReLU


class CBN(nn.Module):
    def __init__(self, backbone, neck, detection_head):
        super(CBN, self).__init__()
        self.backbone = build_backbone(backbone)

        in_channels = neck.in_channels
        self.reduce_layer1 = Conv_BN_ReLU(in_channels[0], 128)
        self.reduce_layer2 = Conv_BN_ReLU(in_channels[1], 128)
        self.reduce_layer3 = Conv_BN_ReLU(in_channels[2], 128)
        self.reduce_layer4 = Conv_BN_ReLU(in_channels[3], 128)

        self.fpem1 = build_neck(neck)
        self.fpem2 = build_neck(neck)

        self.det_head = build_head(detection_head)

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        #return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')
        return F.interpolate(x, size=(H // scale, W // scale), mode='bilinear')

    def forward(self,
                imgs,
                gt_texts=None,
                gt_kernels=None,
                gt_distmap=None,
                training_masks=None,
                gt_instances=None,
                gt_bboxes=None,
                img_metas=None,
                cfg=None):
        outputs = dict()

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            start = time.time()

        # backbone
        f = self.backbone(imgs)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(backbone_time=time.time() - start))
            start = time.time()

        # reduce channel
        f1 = self.reduce_layer1(f[0])
        f2 = self.reduce_layer2(f[1])
        f3 = self.reduce_layer3(f[2])
        f4 = self.reduce_layer4(f[3])

        # FPEM
        f1_1, f2_1, f3_1, f4_1 = self.fpem1(f1, f2, f3, f4)
        f1_2, f2_2, f3_2, f4_2 = self.fpem2(f1_1, f2_1, f3_1, f4_1)

        # FFM
        f1 = f1_1 + f1_2
        f2 = f2_1 + f2_2
        f3 = f3_1 + f3_2
        f4 = f4_1 + f4_2
        f2 = self._upsample(f2, f1.size())
        f3 = self._upsample(f3, f1.size())
        f4 = self._upsample(f4, f1.size())
        f = torch.cat((f1, f2, f3, f4), 1)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(neck_time=time.time() - start))
            start = time.time()

        # detection
        det_out = self.det_head(f)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(det_head_time=time.time() - start))
            start = time.time()

        if self.training:
            det_out[0] = self._upsample(det_out[0], imgs.size())  # init seg results
            det_out[1] = self._upsample(det_out[1], imgs.size())  # distance map branch
            det_out[2] = self._upsample(det_out[2], imgs.size())  # enhanced seg results
            det_loss = self.det_head.loss(det_out, gt_texts, gt_kernels, gt_distmap,
                                          training_masks, gt_instances,
                                          gt_bboxes)
            outputs.update(det_loss)
        else:
            # region / kernel / region distance map 
            det_out_mix = torch.cat([det_out[2], det_out[1][:, :1, :, :]], 1)
            det_out_mix = self._upsample(det_out_mix, imgs.size(), 1) # shape: (n, 3, h, w)
            score = torch.sigmoid(det_out_mix[:, :1, :, :])
            kernels = torch.gt(det_out_mix[:, :2, :, :], 0)
            text_mask = kernels[:, :1, :, :].float()
            kernels[:, 1:, :, :] = kernels[:, 1:, :, :]*text_mask
            dist = det_out_mix[:, 2:, :, :]*text_mask
            det_out_mix = torch.cat([score, kernels.float(), dist], 1)
            if not self.training and cfg.report_speed:
                torch.cuda.synchronize()
                outputs.update(dict(prepare_time=time.time() - start))

            det_res = self.det_head.get_results(det_out_mix, img_metas, cfg)
            outputs.update(det_res)
        return outputs
