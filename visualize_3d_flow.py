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
# Load data
for i, data_blob in enumerate(dataloader):
      (inputs, targets,_ ) = [x.cuda() for x in data_blob]
      targets = targets.cpu().numpy()
      img1 = targets[0]
      break

x, y, z = np.meshgrid(np.arange(344), np.arange(344), np.arange(127), indexing='ij')
u, v, w = img1[..., 0], img1[..., 1], img1[..., 2]

every_xth = 5
x_sub = x[::every_xth, ::every_xth, ::every_xth]
y_sub = y[::every_xth, ::every_xth, ::every_xth]
z_sub = z[::every_xth, ::every_xth, ::every_xth]
u_sub = u[::every_xth, ::every_xth, ::every_xth]
v_sub = v[::every_xth, ::every_xth, ::every_xth]
w_sub = w[::every_xth, ::every_xth, ::every_xth]

mlab.figure()
mlab.quiver3d(x_sub, y_sub, z_sub, u_sub, v_sub, w_sub)
mlab.view(azimuth=0, elevation=90, roll=90)
mlab.show()
