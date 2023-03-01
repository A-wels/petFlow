# creating 3d plot using matplotlib 
# in python
  
  
# importing required libraries
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from core.dataset3dPET import PETDataset3D
import numpy as np
import torch
from tqdm import tqdm
from mayavi import mlab
dataset = PETDataset3D()

batch_size=1
# Create the data loader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, )
img1: np.ndarray
img2: np.ndarray
# Load data
for i, data_blob in enumerate(dataloader):
      (inputs, targets,_ ) = [x.cuda() for x in data_blob]
      inputs = inputs.cpu().numpy()
      img1, img2 = inputs[0]
      print(img1.shape)


      break

mlab.contour3d(img1, colormap="jet", opacity=0.4, contours=30)
mlab.show()