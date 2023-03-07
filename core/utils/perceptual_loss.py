import torch.nn as nn
import torch
import torchvision.models.vgg as vgg

# perceptual loss based on vgg19
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

    def forward(self, pred_flow, gt_flow):
        # load the vgg model
        vgg_model = vgg.vgg19(pretrained=True)
        vgg_features = vgg_model.features

        # set for evaluation
        for param in vgg_features.parameters():
            param.requires_grad = False

        # move to gpu
        vgg_features = vgg_features.cuda()

        # add third channel to flow data
        pred_flow = torch.cat((pred_flow, pred_flow[:, 0:1, :, :]), dim=1)
        gt_flow = torch.cat((gt_flow, gt_flow[:, 0:1, :, :]), dim=1)
        
        # features for predicted flow
        pred_flow_features = vgg_features(pred_flow)

        # get the features of the ground truth flow
        gt_flow_features = vgg_features(gt_flow)

        # calculate the perceptual loss
        perceptual_loss = torch.norm(pred_flow_features - gt_flow_features, p=2, dim=1).mean()
        return perceptual_loss
