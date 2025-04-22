import numpy as np
import math
import matplotlib.pyplot as plt
import time

from baxter import Baxter
from config import SimulationConfig
from dataset import gen_dataset

sim_config_path = "../config/config.toml"
sim_config = SimulationConfig(sim_config_path)

dof = sim_config.dof
alpha_mat = np.identity(dof)
beta_mat = np.identity(dof)
baxter = Baxter(dof, sim_config.alpha_mat, sim_config.beta_mat)

joint_lim_min = np.array([-1.7016, -2.147, -3.0541, -0.05, -3.059, -1.571, -3.059])
joint_lim_max = np.array([1.7016, 1.047, 3.0541, 2.618, 3.059, 2.094, 3.059])
joint_lim_min = joint_lim_min[0:dof]
joint_lim_max = joint_lim_max[0:dof]

gen_dataset(sim_config.num_data, baxter, sim_config.dt, sim_config.T, dof, sim_config.D, sim_config.scaling, sim_config.period, joint_lim_min, joint_lim_max, sim_config.deviation, sim_config.dataset_filename)

