import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import time
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import deepxde as dde
from fourier import FNO1d

class DeepONetProjected(nn.Module):
    # m is typically set to nD*dof*3, dim_x is usually 1, and proj1, proj2 specify the projection of the linear layers. 
    def __init__(self, dim_x, hidden_size, num_layers, n_input_channel, n_output_channel, projection_size, grid, dof, nD):
        super().__init__()
        self.dof = dof
        self.nD = nD
        width = 256
        self.grid = grid
        inputArr = [hidden_size]*num_layers
        outputArr = [hidden_size]*num_layers
        inputArr[0] = n_input_channel
        outputArr[0] = dim_x
        self.deeponet = dde.nn.DeepONetCartesianProd(inputArr, outputArr, "relu", "Glorot normal")
        self.linear1 = torch.nn.Linear(n_input_channel, n_output_channel)

    def forward(self, x):
        y = self.deeponet(x)
        y = self.linear1(y)
        return y

class FNOProjected(nn.Module):
    def __init__(self, dim_x, hidden_size, modes, input_channel, output_channel, dof, nD):
        super().__init__()
        self.dof = dof
        self.nD = nD
        self.fno = FNO1d(modes, hidden_size, dim_x, dim_x)
        self.linear1 = torch.nn.Linear(input_channel, output_channel)

    def forward(self, x):
        x = x.reshape((x.shape[0], x.shape[1], 1))
        y = self.fno(x)
        y = y.reshape((y.shape[0], y.shape[1]))
        y = self.linear1(y)
        return y
        
def ml_predictor_deeponet(_dt, state, controls, model):
    # The conversion time here actually can be expensive to go from CPU to GPU and back.
    dof = model.dof
    nD = model.nD
    inputs = np.zeros((controls.shape[0], dof*3))
    inputs[:, :2*dof] = np.tile(state, controls.shape[0]).reshape(controls.shape[0], 2*dof)
    inputs[:, 2*dof:] = controls
    inputs = inputs.reshape((1, controls.shape[0]*dof*3)).astype(np.float32)
    inputs = torch.from_numpy(inputs).cuda()
    with torch.no_grad():
    	output = model((inputs, model.grid)).detach().cpu().numpy().reshape(nD, 2*dof)
    return output[-1], output

def ml_predictor_fno(_dt, state, controls, model):
    # The conversion time here actually can be expensive to go from CPU to GPU and back.
    dof = model.dof
    nD = model.nD
    inputs = np.zeros((controls.shape[0], dof*3))
    inputs[:, :2*dof] = np.tile(state, controls.shape[0]).reshape(controls.shape[0], 2*dof)
    inputs[:, 2*dof:] = controls
    inputs = inputs.reshape((1, controls.shape[0]*dof*3, 1)).astype(np.float32)
    inputs = torch.from_numpy(inputs).cuda()
    with torch.no_grad():
    	output = model(inputs).detach().cpu().numpy().reshape(nD, 2*dof)
    return output[-1], output

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

class LSTMNet(nn.Module):
    def __init__(self, hidden_size, num_layers, n_input_channel, n_output_channel, dof, nD):
        super(LSTMNet, self).__init__()
        self.dof = dof
        self.nD = nD
        self.rnn = nn.LSTM(n_input_channel, hidden_size, num_layers, batch_first=True)
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
    return output[-1], output

class FNOGRUNet(nn.Module):
    def __init__(self, dim_x, num_layers, fno_hidden_size, gru_hidden_size, modes, input_channel, output_channel, dof, nD):
        super().__init__()
        self.dof = dof
        self.nD = nD
        self.fno = FNO1d(modes, fno_hidden_size, dim_x, dim_x)
        self.rnn = nn.GRU(3*self.dof, gru_hidden_size, num_layers, batch_first=True)
        self.linear1 = torch.nn.Linear(gru_hidden_size, output_channel)

    def forward(self, x):
        x = x.reshape((x.shape[0], x.shape[1], 1))
        y = self.fno(x)
        y = y.reshape((y.shape[0], self.nD, 3*self.dof))
        y, _ = self.rnn(y)
        return self.linear1(y)
 
class DeepONetGRUNet(nn.Module):
    def __init__(self, dim_x, hidden_size, num_layers, n_input_channel, n_output_channel, projection_size, grid, dof, nD):
        super().__init__()
        self.dof = dof
        self.nD = nD
        self.grid = grid
        inputArr = [hidden_size]*num_layers
        outputArr = [hidden_size]*num_layers
        inputArr[0] = n_input_channel
        outputArr[0] = dim_x
        self.deeponet = dde.nn.DeepONetCartesianProd(inputArr, outputArr, "relu", "Glorot normal")
        self.rnn = nn.GRU(n_input_channel, gru_hidden_size, num_layers, batch_first=True)
        self.linear1 = torch.nn.Linear(gru_hidden_size, n_output_channel)

    def forward(self, x):
        op_out = self.deeponet(x)
        y, _ = self.rnn(y)
        return self.linear1(y) +op_out
 
