import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from flowConv2d import FlowNetS
from flowConv2d import OpticalFlowDataset
from utils.EPELoss import EPELoss
import numpy as np

def validate(model, split='validation'):
    model.eval()
    dataset = OpticalFlowDataset(split=split)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, )
    epe_list = []
    criterion = EPELoss()
    with torch.no_grad():
        for i, data_blob in tqdm(enumerate(dataloader), total=len(dataloader)):
            (inputs, targets,_ ) = [x.cuda() for x in data_blob]            
            outputs = model(inputs)
            # use the first output for loss
            output_cropped = outputs[0]
            epe = criterion(output_cropped, targets)
            epe_list.append(epe.detach().cpu())
    print(len(epe_list))
    epe = np.mean(epe_list)
    # px1: percentage of pixels with EPE < 1
    px1 = np.mean(np.array(epe_list) < 1)
    # px3: percentage of pixels with EPE < 3
    px3 = np.mean(np.array(epe_list) < 3)
    # px5: percentage of pixels with EPE < 5
    px5 = np.mean(np.array(epe_list) < 5)
    print("EPE: {:.3f}, px1: {:.3f}, px3: {:.3f}, px5: {:.3f}".format(epe, px1, px3, px5))
    return epe, px1, px3, px5

if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser(description='PyTorch FlowNetS Training')
    parser.add_arguemnt('--mode', type=str, default='validation', help='validation or test')
    args = parser.parse_args()
    mode = args.mode
    assert(mode in ['validation', 'test'])

    
    model = FlowNetS()
    model = model.cuda()
    model = nn.DataParallel(model, [0,1])
    validate(model, split=mode)