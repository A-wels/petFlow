from math import sqrt
import torch
import sys
import os
from numpy.testing import assert_approx_equal
sys.path.append(os.getcwd())
from core.utils.EPELoss import EPELoss

# class for testing nn.Module EPELoss
class TestEPELoss:
    criterion = EPELoss()

    # same values
    def test_one(self):
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        y = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        loss = self.criterion(x, y)
        assert loss == 0.0

    # difference of one
    def test_two(self):
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        y = torch.tensor([[1.0, 2.0, 4.0], [4.0, 5.0, 7.0], [7.0, 8.0, 10.0]])
        loss = self.criterion(x, y)
        assert loss == 1.0

    def test_three(self):
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        y = torch.tensor([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]])
        loss = self.criterion(x, y)
        assert_approx_equal(loss, 3.464, significant=4)

    def test_single(self):
        x = torch.tensor([[1.0, 2.0, 3.0]])
        y = torch.tensor([[3.0, 4.0, 5.0]])
        loss = self.criterion(x, y)
        assert_approx_equal(loss, 3.464, significant=4)

    def test_2d(self):
        x = torch.tensor([[1.0, 1.0]])
        y = torch.tensor([[2.0, 2.0]])
        loss = self.criterion(x, y)
        assert_approx_equal(loss,1.414, significant=4)

    def test_2d_with_different_distances(self):
        x = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        y = torch.tensor([[1.0, 2.0], [2.0, 5.0]])
        target_loss = (sqrt(1.0) + sqrt(1.0 + 4.0**2))/2.0
        loss = self.criterion(x, y)
        assert_approx_equal(loss,target_loss, significant=4)
        
    def test_zeroes(self):
        x = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        y = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        loss = self.criterion(x, y)
        assert_approx_equal(loss,0.0)

