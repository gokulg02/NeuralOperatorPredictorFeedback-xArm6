import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import time
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import deepxde as dde

class DeepONetProjected(nn.Module):
    # m is typically set to nD*dof*3, dim_x is usually 1, and proj1, proj2 specify the projection of the linear layers. 
    def __init__(self, dim_x, hidden_size, num_layers, n_input_channel, n_output_channel, projection_size, grid):
        super().__init__()
        width = 256
        self.grid = grid
        inputArr = [hidden_size]*num_layers
        outputArr = [hidden_size]*num_layers
        inputArr[0] = n_input_channel
        outputArr[0] = dim_x
        self.deeponet = dde.nn.DeepONetCartesianProd(inputArr, outputArr, "relu", "Glorot normal").cuda()
        self.linear1 = torch.nn.Linear(m, projection_size)
        self.linear2 = torch.nn.Linear(projection_size, n_output_channel)

    def forward(self, x):
        y = self.deeponet(x)
        y = self.linear1(y)
        return self.linear2(y)


class GRUNet(nn.Module):
    def __init__(self, hidden_size, num_layers, n_input_channel, n_output_channel, dof, nD):
        super(GRUNet, self).__init__()
        self.dof = dof
        self.nD = nD
        self.rnn = nn.GRU(n_input_channel, hidden_size, num_layers, batch_first=True)
        self.projection = nn.Linear(hidden_size, n_output_channel)

    def forward(self, x: torch.Tensor):
        output, (_) = self.rnn(x)
        x = self.projection(output)
        return x
        
def ml_predictor_rnn(_dt, state, controls, model):
    nD = model.nD
    dof = model.dof
    inputs = np.zeros((controls.shape[0], dof*3))
    inputs[:, :2*dof] = np.tile(state, controls.shape[0]).reshape(controls.shape[0], 2*dof)
    inputs[:, 2*dof:] = controls
    inputs = inputs.reshape((1, controls.shape[0], dof*3)).astype(np.float32)
    inputs = torch.from_numpy(inputs).to(next(model.parameters()).device)
    with torch.no_grad():
        output = model(inputs).detach().cpu().numpy().reshape(nD, 2*dof)
        x_vals = inputs[0].reshape(nD, 3*dof)
        states = x_vals[0, 0:2*dof].detach().cpu().numpy()
        controls = x_vals[:, 2*dof:].detach().cpu().numpy()
    return output[-1], output
