import numpy as np
import math
import matplotlib.pyplot as plt

# only need the imports on lines 6 and 7 if using ML predictors
import torch
from models import GRUNet, LSTMNet, DeepONetProjected, FNOProjected, FNOGRUNet, ml_predictor_rnn, ml_predictor_deeponet, ml_predictor_fno
from config import SimulationConfig, ModelConfig
from baxter import Baxter

# Simulates the system
def simulate_system(baxter, x0, qdes, qdesdot, qdesddot, dt, T, D, dof, predictor_func, model, randomize=False):
    t = np.arange(0, T, dt)
    q_vals = np.zeros((len(t), len(x0)))
    nD = int(round(D/dt))
    controls = np.zeros((len(t)+nD, dof))
    predictors = np.zeros((len(t), nD, len(x0)), dtype=np.float32)
    q_vals[0] = x0
    # Setup initial controllers. This matters - ALOT. 
    controllerStatic = baxter.compute_control(q_vals[0], q_vals[0, :dof], np.zeros(dof), np.zeros(dof))
    controls[0:nD, :] = controllerStatic
    for i in range(1, len(t)):
        if i % 100 == 0:
            print(i, "/", len(t))
        if i > nD:
            if randomize:
                prediction, predictions_arr = predictor_func(dt, q_vals[i-1], controls[i-1:i-1+nD], model) 
                prediction = prediction + np.random.uniform(-0.1, 0.1, 2*dof)
            else:
                prediction, predictions_arr = predictor_func(dt, q_vals[i-1], controls[i-1:i-1+nD], model) 
            controls[i-1+nD] = baxter.compute_control(prediction, qdes[i-1+nD], qdesdot[i-1+nD], qdesddot[i-1+nD])
            if predictions_arr is not None:
                predictors[i] = predictions_arr
        else:
            controls[i-1+nD] = baxter.compute_control(q_vals[i-1], qdes[i-1+nD], qdesdot[i-1+nD], qdesddot[i-1+nD])
            predictors[i] = np.tile(q_vals[i-1], nD).reshape(nD, len(q_vals[i-1]))
        #controls[i-1+nD] = baxter.compute_control(prediction+ np.random.uniform(-0.5, 0.5, 2*dof), qdes[i-1+nD], qdesdot[i-1+nD], qdesddot[i-1+nD])
        q_vals[i] =  baxter.saturate_joints(q_vals[i-1] + dt*baxter.step_q(q_vals[i-1], controls[i-1]))
    return q_vals, controls, predictors

# Generate a trajecotry following a sinusoid
def generate_trajectory(joint_lim_min, joint_lim_max, scaling, period, dt, T, D, y_shift):
    t = np.arange(0, T+D, dt)
    q = np.zeros((len(t), len(joint_lim_min)))
    qd= np.zeros((len(t), len(joint_lim_min)))
    qdd = np.zeros((len(t), len(joint_lim_min)))
    for i in range(len(t)):
        # ensures positive q so we are in the joints
        q[i] = [math.sin(t[i]*period)*scaling ]*len(joint_lim_min) + y_shift
        qd[i] = [math.cos(t[i]*period)*scaling*period]*len(joint_lim_min)
        qdd[i] = [-math.sin(t[i]*period)*scaling*period**2] * len(joint_lim_min)
        if (np.maximum(np.minimum(q[i], joint_lim_max), joint_lim_min) != q[i]).any():
            raise Exception("Trajectory not feasible")
    return q, qd, qdd

### LOAD CONFIG ### 
# Config can be loaded through config.dat or specified for this file in particular using the settings
# below. 
sim_config_path = "../config/config.toml"
sim_config = SimulationConfig(sim_config_path)

model_config_path = "../config/fnogru.toml"
model_config = ModelConfig(model_config_path)

torch.set_default_device(model_config.device_name)


# Baxter parameters
dof = sim_config.dof
alpha_mat = np.identity(dof)
beta_mat = np.identity(dof)
baxter = Baxter(dof, alpha_mat, beta_mat)

joint_lim_min = np.array([-1.7016, -2.147, -3.0541, -0.05, -3.059, -1.571, -3.059])
joint_lim_max = np.array([1.7016, 1.047, 3.0541, 2.618, 3.059, 2.094, 3.059])
joint_lim_min = joint_lim_min[0:dof]
joint_lim_max = joint_lim_max[0:dof]

# Setup initial condition and desired trajectory
init_cond = (joint_lim_max + joint_lim_min) / 2.0 + np.random.uniform(-0.05, 0.05, dof)
y_shift = init_cond
qdes, qd_des, qdd_des = generate_trajectory(joint_lim_min, joint_lim_max, sim_config.scaling, sim_config.period, sim_config.dt, sim_config.T, sim_config.D, y_shift)
init_cond = np.array([init_cond, np.zeros(dof)]).reshape(2*dof)

# Setup type of predictor. If using ML, make sure to modify the parameters below in the match statement to
# load the correct model
predictor_type = "FNO+GRU"
model_filename = model_config.model_filename


match predictor_type:
    case "numerical":
        states, controls, predcitors = simulate_system(baxter, init_cond, qdes, qd_des, qdd_des, sim_config.dt, sim_config.T, sim_config.D, sim_config.dof, baxter.compute_predictors, None, False)
    case "GRU":
        model_config.update_config(input_channel=3*sim_config.dof, output_channel=2*sim_config.dof)
        model = GRUNet(model_config.hidden_size, model_config.num_layers, model_config.input_channel, model_config.output_channel, sim_config.dof, sim_config.nD)
        model.load_state_dict(torch.load("../models/" + model_config.model_filename, weights_only=True))
        model.to(model_config.device)
        states, controls, predcitors = simulate_system(baxter, init_cond, qdes, qd_des, qdd_des, sim_config.dt, sim_config.T, sim_config.D, dof, ml_predictor_rnn, model, False)
    case "LSTM":
        model_config.update_config(input_channel=3*sim_config.dof, output_channel=2*sim_config.dof)
        model = LSTMNet(model_config.hidden_size, model_config.num_layers, model_config.input_channel, model_config.output_channel, sim_config.dof, sim_config.nD)
        model.load_state_dict(torch.load("../models/" + model_config.model_filename, weights_only=True))
        model.to(model_config.device)
        states, controls, predcitors = simulate_system(baxter, init_cond, qdes, qd_des, qdd_des, sim_config.dt, sim_config.T, sim_config.D, dof, ml_predictor_rnn, model, False)
    case "DeepONet":
        spatial = np.arange(0, sim_config.D, sim_config.dt/(3*sim_config.dof)).astype(np.float32)
        grid=torch.from_numpy(spatial.reshape((len(spatial), 1))).to(sim_config.device)
        model_config.update_config(input_channel=grid.shape[0], output_channel=sim_config.nD*2*sim_config.dof)
        model = DeepONetProjected(model_config.dim_x, model_config.hidden_size, model_config.num_layers, model_config.input_channel, model_config.output_channel, model_config.projection_width, grid, sim_config.dof, sim_config.nD)
        model.load_state_dict(torch.load("../models/" + model_config.model_filename, weights_only=True))
        model.to(model_config.device)
        states, controls, predcitors = simulate_system(baxter, init_cond, qdes, qd_des, qdd_des, sim_config.dt, sim_config.T, sim_config.D, dof, ml_predictor_deeponet, model, False)
    case "FNO":
        model_config.update_config(input_channel=3*sim_config.dof, output_channel=2*sim_config.dof)
        model = FNOProjected(model_config.modes, model_config.num_layers, model_config.hidden_size, model_config.input_channel, model_config.output_channel, sim_config.dof, sim_config.nD)
        model.load_state_dict(torch.load("../models/" + model_config.model_filename, weights_only=True))
        states, controls, predcitors = simulate_system(baxter, init_cond, qdes, qd_des, qdd_des, sim_config.dt, sim_config.T, sim_config.D, dof, ml_predictor_rnn, model, False)
    case "FNO+GRU":
        model_config.update_config(input_channel=3*sim_config.dof, output_channel=2*sim_config.dof)
        model = FNOGRUNet(model_config.fno_num_layers, model_config.gru_num_layers, model_config.fno_hidden_size, model_config.gru_hidden_size, model_config.modes, model_config.input_channel, model_config.output_channel, sim_config.dof, sim_config.nD)
        model.load_state_dict(torch.load("../models/" + model_config.model_filename, weights_only=True))
        states, controls, predcitors = simulate_system(baxter, init_cond, qdes, qd_des, qdd_des, sim_config.dt, sim_config.T, sim_config.D, dof, ml_predictor_rnn, model, False)
    case _:
        raise Exception("Predictor type not supported. Please use numerical, GRU, FNO, DeepONet, LSTM, GRU+DeepONet, FNO+GRU.")

plt.plot(states[:, 0])
plt.plot(qdes[:, 0])
plt.savefig("myTraj.pdf")
plt.show()

