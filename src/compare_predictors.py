import numpy as np
import torch
import time
import math

from baxter import Baxter
from models import GRUNet, LSTMNet, DeepONetProjected, FNOProjected, FNOGRUNet, DeepONetGRUNet, ml_predictor_rnn, ml_predictor_deeponet
from config import SimulationConfig, ModelConfig

sim_config_path = "../config/config.toml"
sim_config = SimulationConfig(sim_config_path)

model_config_path_lstm = "../config/lstm.toml"
model_config_lstm = ModelConfig(model_config_path_lstm)

model_config_path_gru = "../config/gru.toml"
model_config_gru = ModelConfig(model_config_path_gru)

model_config_path_deeponet = "../config/deeponet.toml"
model_config_deeponet = ModelConfig(model_config_path_deeponet)

model_config_path_fno = "../config/fno.toml"
model_config_fno = ModelConfig(model_config_path_fno)

model_config_path_fnogru = "../config/fnogru.toml"
model_config_fnogru = ModelConfig(model_config_path_fnogru)

model_config_path_deeponetgru = "../config/deeponetgru.toml"
model_config_deeponetgru = ModelConfig(model_config_path_deeponetgru)

torch.set_default_device(model_config_deeponet.device_name)

model_config_gru.update_config(input_channel=3*sim_config.dof, output_channel=2*sim_config.dof)
model_config_lstm.update_config(input_channel=3*sim_config.dof, output_channel=2*sim_config.dof)
model_config_fno.update_config(input_channel=3*sim_config.dof, output_channel=2*sim_config.dof)
model_config_fnogru.update_config(input_channel=3*sim_config.dof, output_channel=2*sim_config.dof)

names = ["numerical", "gru", "lstm", "deeponet", "fno", "fnogru", "deeponetgru"]
time_steps = [0.1, 0.05, 0.01]
delays = [0.1, 0.5, 1]
predictor_trials = 100
numerical_times = np.zeros((len(time_steps), len(delays)))
gru_times = np.zeros((len(time_steps), len(delays)))
lstm_times = np.zeros((len(time_steps), len(delays)))
deeponet_times = np.zeros((len(time_steps), len(delays)))
fno_times = np.zeros((len(time_steps), len(delays)))
fnogru_times = np.zeros((len(time_steps), len(delays)))
deeponetgru_times = np.zeros((len(time_steps), len(delays)))

for i, dt in enumerate(time_steps):
    for j, D in enumerate(delays):

        dof = sim_config.dof
        nD = int(round(D/dt))
        state = np.zeros(2*dof)
        controls = np.zeros((nD, dof))

        modelGRU = GRUNet(model_config_gru.hidden_size, model_config_gru.num_layers, model_config_gru.input_channel, model_config_gru.output_channel, sim_config.dof, nD)
         
        modelLSTM = LSTMNet(model_config_lstm.hidden_size, model_config_lstm.num_layers, model_config_lstm.input_channel, model_config_lstm.output_channel, sim_config.dof, nD)
         
        spatial = np.arange(0, D, dt/(3*sim_config.dof)).astype(np.float32)
        grid=torch.from_numpy(spatial.reshape((len(spatial), 1))).to(sim_config.device)
        model_config_deeponet.update_config(input_channel=grid.shape[0], output_channel=nD*2*sim_config.dof)
        modelDeepONet = DeepONetProjected(model_config_deeponet.dim_x, model_config_deeponet.hidden_size, model_config_deeponet.num_layers, model_config_deeponet.input_channel, model_config_deeponet.output_channel, grid, sim_config.dof, nD)

        modelFNO = FNOProjected(model_config_fno.hidden_size, model_config_fno.num_layers, model_config_fno.modes, model_config_fno.input_channel, model_config_fno.output_channel, sim_config.dof, nD)

        modelFNOGRU = FNOGRUNet(model_config_fnogru.fno_num_layers, model_config_fnogru.gru_num_layers, model_config_fnogru.fno_hidden_size, model_config_fnogru.gru_hidden_size, model_config_fnogru.modes, model_config_fnogru.input_channel, model_config_fnogru.output_channel, sim_config.dof, nD)

        model_config_deeponetgru.update_config(input_channel=grid.shape[0], output_channel=nD*2*sim_config.dof)
        modelDeepONetGRU = DeepONetGRUNet(model_config_deeponetgru.dim_x, model_config_deeponetgru.deeponet_num_layers, model_config_deeponetgru.gru_num_layers, model_config_deeponetgru.deeponet_hidden_size, model_config_deeponetgru.gru_hidden_size, model_config_deeponetgru.input_channel, model_config_deeponetgru.output_channel, grid, sim_config.dof, nD)

        alpha_mat = np.identity(dof)
        beta_mat = np.identity(dof)
        baxter = Baxter(dof, alpha_mat, beta_mat)

        start_time = time.time()
        for k in range(predictor_trials):
            baxter.compute_predictors(dt, state, controls, None)
        end_time = time.time()
        numerical_times[i][j] = (end_time-start_time)/predictor_trials*1000
        start_time = time.time()
        for k in range(predictor_trials):
            ml_predictor_rnn(dt, state, controls, modelGRU)
        end_time = time.time()
        gru_times[i][j] = (end_time-start_time)/predictor_trials * 1000
        start_time = time.time()
        for k in range(predictor_trials):
            ml_predictor_rnn(dt, state, controls, modelLSTM)
        end_time = time.time()
        lstm_times[i][j] = (end_time-start_time)/predictor_trials * 1000
        start_time = time.time()
        for k in range(predictor_trials):
            ml_predictor_deeponet(dt, state, controls, modelDeepONet)
        end_time = time.time()
        deeponet_times[i][j] = (end_time-start_time)/predictor_trials * 1000
        start_time = time.time()
        for k in range(predictor_trials):
            ml_predictor_rnn(dt, state, controls, modelFNO)
        end_time = time.time()
        fno_times[i][j] = (end_time-start_time)/predictor_trials * 1000
        start_time = time.time()
        for k in range(predictor_trials):
            ml_predictor_rnn(dt, state, controls, modelFNOGRU)
        end_time = time.time()
        fnogru_times[i][j] = (end_time-start_time)/predictor_trials * 1000
        start_time = time.time()
        for k in range(predictor_trials):
            ml_predictor_deeponet(dt, state, controls, modelDeepONetGRU)
        end_time = time.time()
        deeponetgru_times[i][j] = (end_time-start_time)/predictor_trials * 1000
        print("Delay:", delays[j], "dt", time_steps[i])
        print(names)
        print(numerical_times[i][j], gru_times[i][j], lstm_times[i][j], fno_times[i][j], deeponet_times[i][j], fnogru_times[i][j], deeponetgru_times[i][j])
