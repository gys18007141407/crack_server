import torch
import torch.nn as nn
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1) # [batch, total_pixel]
        target = target.contiguous().view(target.shape[0], -1)    # [batch, total_pixel]
        num = torch.sum(torch.mul(predict, target), dim=1) + 1			# [batch, total_pixel]  sum ---> [batch]
        den = torch.sum(predict.pow(2) + target.pow(2), dim=1) + 1		# [batch, total_pixel]  sum ---> [batch]
        return 1 - num / den

