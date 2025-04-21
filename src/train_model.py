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
from config import SimulationConfig, ModelConfig

### LOAD CONFIG ### 
# Config can be loaded through config.dat or specified for this file in particular using the settings
# below. 
sim_config_path = "../config/config.toml"
sim_config = SimulationConfig(sim_config_path)

model_config_path = "../config/gru.toml"
model_config = ModelConfig(model_config_path)

# Ensure that our data and model will live on the same cpu/gpu device. 
assert(model_config.device == sim_config.device)
torch.set_default_device(sim_config.device_name)

inputs = np.load("../datasets/inputs" + sim_config.dataset_filename+".npy").astype(np.float32)
outputs = np.load("../datasets/outputs" + sim_config.dataset_filename + ".npy").astype(np.float32)

x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, test_size=sim_config.test_size, random_state=sim_config.random_state)
x_train = torch.from_numpy(x_train).to(sim_config.device)
x_test = torch.from_numpy(x_test).to(sim_config.device)
y_train = torch.from_numpy(y_train).to(sim_config.device)
y_test = torch.from_numpy(y_test).to(sim_config.device)

trainData = DataLoader(TensorDataset(x_train, y_train), batch_size=sim_config.batch_size, shuffle=True, generator=torch.Generator(device=sim_config.device))
testData = DataLoader(TensorDataset(x_test, y_test), batch_size=sim_config.batch_size, shuffle=False, generator=torch.Generator(device=sim_config.device))

match model_config.model_type:
    case "GRU":
        # Update Model Configs to for compatability
        model_config.update_config(input_channel=3*sim_config.dof, output_channel=2*sim_config.dof)
        model = GRUNet(model_config.hidden_size, model_config.num_layers, model_config.input_channel, model_config.output_channel, sim_config.dof, sim_config.nD)
    case "DeepONet":
        spatial = np.arange(0, sim_config.D, sim_config.dt/(3*sim_config.dof)).astype(np.float32)
        grid=torch.from_numpy(spatial.reshape((len(spatial), 1))).to(sim_config.device)
        model_config.update_config(input_channel=grid_shape[0], output_channel=sim_config.nD*2*sim_config.dof)
        model = DeepONetProjected(model_config.dim_x, model_config.num_layers, model_config.n_input_channel, model_config.n_output_channel, model_config.projection_width, grid)
    case _:
        raise Exception("Model type not supported. Please use GRU, FNO, DeepONet, LSTM, GRU+DeepONet, GRU+FNO.")

model.to(sim_config.device)

# Saves the best test loss model in the locations "models/model_filename"
model, train_loss_arr, test_loss_arr = model_trainer(model, trainData, testData, model_config.epochs, sim_config.batch_size, model_config.gamma, model_config.learning_rate, model_config.weight_decay, model_config.model_type, model_config.model_filename)
evaluate_model(model, model_config.model_type, train_loss_arr, test_loss_arr)
