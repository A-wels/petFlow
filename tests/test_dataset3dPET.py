import torch
import sys
import os

sys.path.append(os.getcwd())
from core.utils.EPELoss import EPELoss
from core.dataset3dPET import PETDataset3D

# class for testing nn.Module EPELoss
class TestDatasetPET3d:
    dataset = PETDataset3D()

    def test_load_data(self):
     # Create the data loader
        batch_size = 5
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True, drop_last=True )

        # Train the model
        for i, data_blob in enumerate(dataloader):
            (inputs, targets,_ ) = [x.cuda() for x in data_blob]

            # input shape: (batch_size, 2, 344, 344, 127)
            assert(inputs.shape == (batch_size, 2, 344, 344, 127))

            # flow shape: (batch_size, 344, 344, 127, 3)
            assert(targets.shape == (batch_size, 344, 344, 127 ,3))
            break

