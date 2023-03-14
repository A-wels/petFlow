import torch.nn as nn
import torch

class EPELoss(nn.Module):
    def __init__(self):
        super(EPELoss, self).__init__()

    def forward(self, pred_flow, gt_flow):
        # calculate the epe
        epe = torch.norm(pred_flow - gt_flow, p=2, dim=1).mean()
        return epe

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, pred_flow, gt_flow):
        # calculate the MSE loss
        mse = torch.mean((pred_flow - gt_flow)**2)
        return mse
        