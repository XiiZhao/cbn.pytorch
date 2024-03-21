import torch
import torch.nn as nn


class RatioLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(RatioLoss, self).__init__()
        self.loss_weight = loss_weight
        self.relu = torch.nn.ReLU()

    def forward(self, input, target, mask, reduce=True):
        batch_size = input.size(0)
        input = self.relu(input)

        input = input.contiguous().view(batch_size, -1)
        target = target.contiguous().view(batch_size, -1).float()
        mask = mask.contiguous().view(batch_size, -1).float()

        input = input * mask
        target = target * mask
       
        ratio = torch.max(input, target) / (torch.min(input, target) + 1e-5)

        ratio[ratio<1.0] = 1.0
        loss = torch.log(ratio)

        #a = torch.sum(input * target, dim=1)
        #b = torch.sum(input * input, dim=1) + 0.001
        #c = torch.sum(target * target, dim=1) + 0.001
        #d = (2 * a) / (b + c)
        #loss = 1 - d

        loss = self.loss_weight * loss

        if reduce:
            loss = torch.mean(loss)

        return loss
