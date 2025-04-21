import torch
import time
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from models import GRUNet, DeepONetProjected
from trainer import model_trainer, evaluate_model

# Set parameters
# Baxter
dof = 3
T = 10
period = 1
dt = 0.1
scaling = 0.1
t = np.arange(0, T, dt)
# Handles delays up to any time essentially
D = 0.5
nD = int(round(D/dt))

# File paths
dataset_filename = "testDataset"
model_filename = "testGRU"

# Set cuda or cpu device. Replace "cuda" with "cpu" to switch. MACOS uses cpu here even with builtin gpu. 
torch.set_default_device('cuda')
device = torch.device("cuda")

# Set random state
random_state = 1

# Model params
# Model type options: "GRU", "LSTM", "DeepONet", "FNO", "GRU+DeepONet", "GRU+FNO"
# Set model neuron_size, layers, etc. below if you'd like. Otherwise will use a default
model_type = "GRU"
epochs = 100
test_size = 0.1
batch_size = 2048
gamma = 0.98
learning_rate = 0.0003
weight_decay=1.5e-2

inputs = np.load("../datasets/inputs" + dataset_filename+".npy").astype(np.float32)
outputs = np.load("../datasets/outputs" + dataset_filename + ".npy").astype(np.float32)

x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, test_size=test_size, random_state=random_state)
x_train = torch.from_numpy(x_train).to(device)
x_test = torch.from_numpy(x_test).to(device)
y_train = torch.from_numpy(y_train).to(device)
y_test = torch.from_numpy(y_test).to(device)

trainData = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))
testData = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False, generator=torch.Generator(device=device))

match model_type:
    case "GRU":
        model = GRUNet(64, 5, 3*dof, 2*dof, dof, nD)
    case "DeepONet":
        spatial = np.arange(0, D, dt/(3*dof)).astype(np.float32)
        grid=torch.from_numpy(spatial.reshape((len(spatial), 1))).to(device)
        model = DeepONetProjected(grid.shape[0], 1, 48, nD*2*dof, grid)
    case _:
        raise Exception("Model type not supported. Please use GRU, FNO, DeepONet, LSTM, GRU+DeepONet, GRU+FNO.")

model.to(device)

# Saves the best test loss model in the locations "models/model_filename"
model, train_loss_arr, test_loss_arr = model_trainer(model, trainData, testData, epochs, batch_size, gamma, learning_rate, weight_decay, model_type, model_filename)
evaluate_model(model, model_type, train_loss_arr, test_loss_arr)
