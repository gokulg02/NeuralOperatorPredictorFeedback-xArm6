import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import time
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import deepxde as dde
from neuralop.models import FNO1d

def ml_predictor(state, controls, delay, x, model):
    nx = len(x)
    inputs = np.zeros((nx, 6))
    inputs[:, 0] = np.full(nx, delay)
    inputs[:, 1:4] = np.tile(state, (nx, 1))
    inputs[:, 4:6] = controls
    inputs = inputs.reshape((1, nx, 6)).astype(np.float32)
    inputs = torch.from_numpy(inputs).to(next(model.parameters()).device)
    with torch.no_grad():
        output = model(inputs)
    return output[-1].detach().cpu().numpy()

class DeepONetProjected(nn.Module):
    def __init__(self, dim_x, hidden_size, num_layers, n_input_channel, n_output_channel, grid):
        super().__init__()
        self.grid = grid
        inputArr = [hidden_size]*num_layers
        outputArr = [hidden_size]*num_layers
        inputArr[0] = n_input_channel
        outputArr[0] = dim_x
        self.deeponet = dde.nn.DeepONetCartesianProd(inputArr, outputArr, "relu", "Glorot normal")
        self.out = torch.nn.Sequential(torch.nn.Linear(n_input_channel, 4 * n_output_channel),torch.nn.ReLU(),torch.nn.Linear(4 * n_output_channel, n_output_channel))

    def forward(self, x):
        nD = x.shape[1]
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
        y = self.deeponet((x, self.grid))
        y = self.out(y)
        # Fix back to dataset tensor size
        y = y.reshape(y.shape[0], nD, y.shape[1]//nD)
        return y

class FNOProjected(nn.Module):
    def __init__(self, hidden_size, num_layers, modes, input_channel, output_channel):
        super().__init__()
        self.fno = FNO1d(n_modes_height=modes, n_layers=num_layers, hidden_channels=hidden_size, in_channels=input_channel, out_channels=output_channel)

    def forward(self, x):
        y = self.fno(x.transpose(1, 2))
        return y.transpose(1, 2)
        
class GRUNet(nn.Module):
    def __init__(self, hidden_size, num_layers, n_input_channel, n_output_channel):
        super(GRUNet, self).__init__()
        self.rnn = nn.GRU(n_input_channel, hidden_size, num_layers, batch_first=True)
        self.projection = nn.Linear(hidden_size, n_output_channel)

    def forward(self, x: torch.Tensor):
        output, (_) = self.rnn(x)
        x = self.projection(output)
        return x

class LSTMNet(nn.Module):
    def __init__(self, hidden_size, num_layers, n_input_channel, n_output_channel):
        super(LSTMNet, self).__init__()
        self.rnn = nn.LSTM(n_input_channel, hidden_size, num_layers, batch_first=True)
        self.projection = nn.Linear(hidden_size, n_output_channel)

    def forward(self, x: torch.Tensor):
        output, (_) = self.rnn(x)
        x = self.projection(output)
        return x

class FNOGRUNet(nn.Module):
    def __init__(self, fno_num_layers, gru_num_layers, fno_hidden_size, gru_hidden_size, modes, input_channel, output_channel):
        super().__init__()
        self.fno = FNOProjected(fno_hidden_size, fno_num_layers, modes, input_channel, output_channel)
        self.rnn = nn.GRU(output_channel, gru_hidden_size, gru_num_layers, batch_first=True)
        self.linear1 = torch.nn.Linear(gru_hidden_size, output_channel)

    def forward(self, x):
        fno_out= self.fno(x)
        y, _ = self.rnn(fno_out)
        y = self.linear1(y)
        # time aware neural operator
        return y 
 
class DeepONetGRUNet(nn.Module):
    def __init__(self, dim_x, deeponet_num_layers, gru_num_layers, deeponet_hidden_size, gru_hidden_size, n_input_channel, n_output_channel, grid):
        super().__init__()
        self.grid = grid
        self.deeponet = DeepONetProjected(dim_x, deeponet_hidden_size, deeponet_num_layers, n_input_channel, n_output_channel, grid)
        self.rnn = nn.GRU(3, gru_hidden_size, gru_num_layers, batch_first=True)
        self.linear1 = torch.nn.Linear(gru_hidden_size,3)

    def forward(self, x):
        deeponet_out = self.deeponet(x)
        y, _ = self.rnn(deeponet_out)
        return self.linear1(y)
 
