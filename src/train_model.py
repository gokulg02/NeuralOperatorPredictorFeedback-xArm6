import torch
import time
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torchtune

from models import GRUNet, LSTMNet, DeepONetProjected, FNOProjected, FNOGRUNet, ml_predictor_rnn
from trainer import model_trainer, evaluate_model
from config import SimulationConfig, ModelConfig
from dataset import gen_dataset_dagger
from baxter import Baxter

### LOAD CONFIG ### 
# Config can be loaded through config.dat or specified for this file in particular using the settings
# below. 
sim_config_path = "../config/config.toml"
sim_config = SimulationConfig(sim_config_path)

model_config_path = "../config/fnogru.toml"
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
        ml_predictor = ml_predictor_rnn
    case "LSTM":
        # Update Model Configs to for compatability
        model_config.update_config(input_channel=3*sim_config.dof, output_channel=2*sim_config.dof)
        model = LSTMNet(model_config.hidden_size, model_config.num_layers, model_config.input_channel, model_config.output_channel, sim_config.dof, sim_config.nD)
        ml_predictor = ml_predictor_rnn
    case "DeepONet":
        spatial = np.arange(0, sim_config.D, sim_config.dt/(3*sim_config.dof)).astype(np.float32)
        grid=torch.from_numpy(spatial.reshape((len(spatial), 1))).to(sim_config.device)
        model_config.update_config(input_channel=grid.shape[0], output_channel=sim_config.nD*2*sim_config.dof)
        model = DeepONetProjected(model_config.dim_x, model_config.hidden_size, model_config.num_layers, model_config.input_channel, model_config.output_channel, model_config.projection_width, grid, sim_config.dof, sim_config.nD)
    case "FNO":
        model_config.update_config(input_channel=3*sim_config.dof, output_channel=2*sim_config.dof)
        model = FNOProjected(model_config.hidden_size, model_config.num_layers, model_config.modes, model_config.input_channel, model_config.output_channel, sim_config.dof, sim_config.nD)
    case "FNO+GRU":
        model_config.update_config(input_channel=3*sim_config.dof, output_channel=2*sim_config.dof)
        model = FNOGRUNet(model_config.fno_num_layers, model_config.gru_num_layers, model_config.fno_hidden_size, model_config.gru_hidden_size, model_config.modes, model_config.input_channel, model_config.output_channel, sim_config.dof, sim_config.nD)
    case _:
        raise Exception("Model type not supported. Please use GRU, FNO, DeepONet, LSTM, GRU+DeepONet, GRU+FNO.")

model.to(sim_config.device)
print("Model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
# Saves the best test loss model in the locations "models/model_filename"
# Saves the loss function curve to this directory
model, train_loss_arr, test_loss_arr = model_trainer(model, trainData, testData, model_config.epochs, sim_config.batch_size, model_config.gamma, model_config.learning_rate, model_config.weight_decay, model_config.model_type, model_config.model_filename)

# if sim_config.dagger:
#     dof = sim_config.dof
#     alpha_mat = np.identity(dof)
#     beta_mat = np.identity(dof)
#     baxter = Baxter(dof, sim_config.alpha_mat, sim_config.beta_mat)

#     joint_lim_min = np.array([-1.7016, -2.147, -3.0541, -0.05, -3.059, -1.571, -3.059])
#     joint_lim_max = np.array([1.7016, 1.047, 3.0541, 2.618, 3.059, 2.094, 3.059])
#     joint_lim_min = joint_lim_min[0:dof]
#     joint_lim_max = joint_lim_max[0:dof]

#     print("Base trainining finished. Commencing dynamic training")
#     # Dagger training loop
#     for i in range(model_config.iters_dag):
#         print("Completed Dag iter", i, "/", model_config.iters_dag)
#         inputsDag, outputsDag = gen_dataset_dagger(sim_config.num_data_dagger, baxter, sim_config.dt, sim_config.T, dof, sim_config.D, sim_config.scaling, sim_config.period, joint_lim_min, joint_lim_max, sim_config.deviation_dagger, ml_predictor, model)
#         x_train_dag, x_test_dag, y_train_dag, y_test_dag = train_test_split(inputsDag, outputsDag, test_size=sim_config.test_size, random_state=sim_config.random_state)
#         x_train_dag = torch.from_numpy(x_train_dag).to(sim_config.device)
#         x_test_dag = torch.from_numpy(x_test_dag).to(sim_config.device)
#         y_train_dag = torch.from_numpy(y_train_dag).to(sim_config.device)
#         y_test_dag = torch.from_numpy(y_test_dag).to(sim_config.device)
#         x_train = torch.concat([x_train, x_train_dag])
#         x_test = torch.concat([x_test, x_test_dag])
#         y_train = torch.concat([y_train, y_train_dag])
#         y_test = torch.concat([y_test, y_test_dag])
 
#         trainDataDag = DataLoader(TensorDataset(x_train, y_train), batch_size=sim_config.batch_size, shuffle=True, generator=torch.Generator(device=sim_config.device))
#         testDataDag = DataLoader(TensorDataset(x_test, y_test), batch_size=sim_config.batch_size, shuffle=False, generator=torch.Generator(device=sim_config.device))
        
#         model, train_loss_arr, test_loss_arr = model_trainer(model, trainDataDag, testDataDag, model_config.epochs_dag, sim_config.batch_size_dagger, model_config.gamma_dag, model_config.learning_rate_dag, model_config.weight_decay_dag, model_config.model_type, model_config.model_filename)
evaluate_model(model, model_config.model_type, train_loss_arr, test_loss_arr)
