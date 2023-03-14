from collections import namedtuple
import torch.nn as nn
import torch
import torchvision.models.vgg as vgg

from core.utils.EPELoss import MSELoss

# perceptual loss based on vgg19 and https://towardsdatascience.com/pytorch-implementation-of-perceptual-losses-for-real-time-style-transfer-8d608e2e9902
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # define loss module 
        self.vgg_loss = LossVGG()
    def forward(self, pred_flow, gt_flow):


        # add third channel to flow data
        pred_flow = torch.cat((pred_flow, pred_flow[:, 0:1, :, :]), dim=1)
        gt_flow = torch.cat((gt_flow, gt_flow[:, 0:1, :, :]), dim=1)
        
        # features for predicted flow
        pred_flow_features = self.vgg_loss(pred_flow)

        # features for ground truth flow
        with torch.no_grad():
            gt_flow_features = self.vgg_loss(gt_flow)
        CONTENT_WEIGHT = 1
        mse_loss = MSELoss()
        loss = CONTENT_WEIGHT * mse_loss(pred_flow_features[2], gt_flow_features[2]) 
        return loss

LossOutput = namedtuple(
    "LossOutput", ["relu1", "relu2", "relu3", "relu4", "relu5"])

class LossVGG(nn.Module):
    """Reference:
    https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
    """

    def __init__(self):
        super(LossVGG, self).__init__()
        self.vgg_layers = vgg.vgg19(pretrained=True).features
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vgg_layers.to(device)
        self.layer_name_mapping = {
            '3': "relu1",
            '8': "relu2",
            '17': "relu3",
            '26': "relu4",
            '35': "relu5",
        }

    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return LossOutput(**output)