import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from core.utils.EPELoss import EPELoss
from core.utils.perceptual_loss import PerceptualLoss
import numpy as np
from core.flowNetS import FlowNetS
from core.dataset_pet import PETDataset

def validate(model, split='validation'):
    model.eval()
    dataset = PETDataset(split=split)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,drop_last=True )
    epe_list = []
    perceptual_list = []
    criterionEPE = EPELoss()
    criterionPerceptual = PerceptualLoss()
    with torch.no_grad():
        for i, data_blob in tqdm(enumerate(dataloader), total=len(dataloader)):
            (inputs, targets,_ ) = [x.cuda() for x in data_blob]            
            output = model(inputs)
            # use the first output for loss
            epe = criterionEPE(output, targets)
            perceptual = criterionPerceptual(output, targets)
            epe_list.append(epe.detach().cpu())
            perceptual_list.append(perceptual.detach().cpu())
    # epe: average end point error
    epe = np.mean(epe_list)
    # perceptual: average perceptual loss
    perceptual = np.mean(perceptual_list)
    # px1: percentage of pixels with EPE < 1
    px1 = np.mean(np.array(epe_list) < 1)
    # px3: percentage of pixels with EPE < 3
    px3 = np.mean(np.array(epe_list) < 3)
    # px5: percentage of pixels with EPE < 5
    px5 = np.mean(np.array(epe_list) < 5)
    print("EPE: {:.3f}, px1: {:.3f}, px3: {:.3f}, px5: {:.3f}, perceptual: {:.3f}".format(epe, px1, px3, px5, perceptual))
    return epe, px1, px3, px5, perceptual

if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser(description='PyTorch FlowNetS Training')
    parser.add_argument('--testing', action="store_true", help='Specify testing dataset testing')
    args = parser.parse_args()
    mode = "testing" if args.testing else "validation"

    
    model = FlowNetS()
    model = model.cuda()
    model = nn.DataParallel(model, [0,1])
    validate(model, split=mode)