import argparse
import random
import sys

from tqdm import tqdm
sys.path.append('core')
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from glob import glob
import os.path as osp
from utils.frame_utils import read_gen
from utils.EPELoss import EPELoss
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn import init
from validate import validate
from core.dataset_pet import PETDataset
from core.flowNetS import FlowNetS


 # main method
if __name__ == '__main__':
    # Create arguments parser and parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=350)
    parser.add_argument('--c', type=bool, default=False, help='load from checkpoint')
    args = parser.parse_args()

    # Create a summary writer
    writer = SummaryWriter(log_dir="logs")

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define model
    model = FlowNetS().to(device)
    
    model = nn.DataParallel(model, [0,1])

    # load from checkpoint if specified
    if args.c:
        model.load_state_dict(torch.load("checkpoints/optical_flow_2d.pt"))

   # model = model.to(device)
    model = model.cuda()
    model.train()

    # Define a loss function
    #criterion = nn.MSELoss()
    criterion = EPELoss()

    # Define an optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # Load your data into a numpy array of shape (num_samples, 2, 344, 344, 172)


    # Create the dataset
    dataset = PETDataset()

    # Create the data loader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, )

    # Define an optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    # Train the model
    epoch_num = args.epochs 
    for epoch in range(epoch_num):
        print("Starting training epoch {} out of {}".format(epoch+1, epoch_num))
        for i, data_blob in tqdm(enumerate(dataloader), total=len(dataloader)):
            (inputs, targets,_ ) = [x.cuda() for x in data_blob]
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            output_cropped = outputs[0]
            loss = criterion(output_cropped, targets)
            # Write log
            writer.add_scalar("Loss/train", loss, epoch)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
        # Print the loss every X epochs
        if (epoch+1) % 1 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epoch_num, loss.item()))

        # Validate every X epochs
        if (epoch+1)%100 == 0:
            epe, px1, px3, px5 = validate(model)
            model.train()
            writer.add_scalar("EPE", epe, epoch)
            writer.add_scalar("px1", px1, epoch)
            writer.add_scalar("px3", px3, epoch)
            writer.add_scalar("px5", px5, epoch)
            
        if (epoch+1) % 350 == 0:
            torch.save(model.state_dict(), 'checkpoints/optical_flow_2d_{}.pt'.format(epoch+1))

    # Save the model
    writer.flush()

