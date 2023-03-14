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
from core.utils.frame_utils import read_gen
from core.utils.EPELoss import EPELoss
from core.utils.perceptual_loss import PerceptualLoss
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn import init
from validate import validate
from core.dataset_pet import PETDataset
from core.flowNetS import FlowNetS


batch_size = 128
 # main method
if __name__ == '__main__':
    # Create arguments parser and parese args
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=350)
    # continue training?
    parser.add_argument('--c', action="store_true", help='Continue training?')
    parser.add_argument('--done_steps', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default="")
    args = parser.parse_args()

    done_steps = args.done_steps

    # Create a summary writer
    writer = SummaryWriter(log_dir=os.path.join("logs",args.log_dir))

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define model and move to device
    model = FlowNetS().to(device)
    
    model = nn.DataParallel(model, [0,1])

    # load from checkpoint if specified
    if args.c:
        model.load_state_dict(torch.load("checkpoints/optical_flow_2d.pt"))

    model = model.cuda()
    model.train()

    # Define a loss function
    #criterion = nn.MSELoss()
    criterion = PerceptualLoss()

    # Define an optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Create the dataset
    dataset = PETDataset()

    # Create the data loader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True )

    # Define an optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    # Train the model
    epoch_num = args.epochs 
    for epoch in range(epoch_num):
        epoch+=done_steps
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
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epoch_num, loss.item()))

        # save the model every X epochs
        if (epoch+1) % 200 == 0:
            torch.save(model.state_dict(), 'checkpoints/optical_flow_2d_{}.pt'.format(epoch+1))
        # Validate every X epochs
        if (epoch+1)%200 == 0:
            epe, px1, px3, px5, perceptual = validate(model)
            model.train()
            writer.add_scalar("EPE", epe, epoch)
            writer.add_scalar("px1", px1, epoch)
            writer.add_scalar("px3", px3, epoch)
            writer.add_scalar("px5", px5, epoch)
            writer.add_scalar("perceptual loss", perceptual, epoch)
            
       

    # Save the model
    torch.save(model.state_dict(), 'checkpoints/optical_flow_2d.pt')
    writer.flush()

