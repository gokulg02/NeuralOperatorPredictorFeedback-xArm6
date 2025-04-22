import numpy as np
import torch
import time
import math

from baxter import Baxter
from models import GRUNet, LSTMNet, DeepONetProjected, ml_predictor_rnn, ml_predictor_deeponet
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

torch.set_default_device(model_config_deeponet.device_name)

model_config_gru.update_config(input_channel=3*sim_config.dof, output_channel=2*sim_config.dof)
model_config_lstm.update_config(input_channel=3*sim_config.dof, output_channel=2*sim_config.dof)
model_config_fno.update_config(input_channel=3*sim_config.dof, output_channel=2*sim_config.dof)

time_steps = [0.1, 0.05, 0.02]
delays = [0.1, 0.5, 1]
predictor_trials = 100
numerical_times = np.zeros((len(time_steps), len(delays)))
gru_times = np.zeros((len(time_steps), len(delays)))
lstm_times = np.zeros((len(time_steps), len(delays)))
deeponet_times = np.zeros((len(time_steps), len(delays)))
fno_times = np.zeros((len(time_steps), len(delays)))

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
        modelDeepONet = DeepONetProjected(model_config_deeponet.dim_x, model_config_deeponet.hidden_size, model_config_deeponet.num_layers, model_config_deeponet.input_channel, model_config_deeponet.output_channel, model_config_deeponet.projection_width, grid, sim_config.dof, nD)

        model = FNOProjected(model_config.dim_x, model_config.hidden_size, model_config.modes, model_config.input_channel, model_config.output_channel, sim_config.dof, nD)

        alpha_mat = np.identity(dof)
        beta_mat = np.identity(dof)
        baxter = Baxter(dof, alpha_mat, beta_mat)

        start_time = time.time()
        for k in range(predictor_trials):
            baxter.compute_predictors(dt, state, controls, None)
        end_time = time.time()
        numerical_times[i][j] = end_time-start_time
        start_time = time.time()
        for k in range(predictor_trials):
            ml_predictor_rnn(dt, state, controls, modelGRU)
        end_time = time.time()
        gru_times[i][j] = end_time-start_time
        start_time = time.time()
        for k in range(predictor_trials):
            ml_predictor_rnn(dt, state, controls, modelLSTM)
        end_time = time.time()
        lstm_times[i][j] = end_time-start_time
        start_time = time.time()
        for k in range(predictor_trials):
            ml_predictor_deeponet(dt, state, controls, modelDeepONet)
        end_time = time.time()
        deeponet_times[i][j] = end_time-start_time
        start_time = time.time()
        for k in range(predictor_trials):
            ml_predictor_fno(dt, state, controls, modelFNO)
        end_time = time.time()
        fno_times[i][j] = end_time-start_time
