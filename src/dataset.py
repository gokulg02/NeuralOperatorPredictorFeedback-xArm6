import numpy as np
import math
import matplotlib.pyplot as plt
import time

from baxter import Baxter
from config import SimulationConfig

# Simulates the system
def simulate_system(baxter, x0, q_des, q_desdot, q_desddot, dt, T, D, dof, predictor_func, model, randomize=False, deviation=0):
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
                prediction = prediction + np.random.uniform(-1*deviation, deviation, 2*dof)
            else:
                prediction, predictions_arr = predictor_func(dt, q_vals[i-1], controls[i-1:i-1+nD], model) 
            controls[i-1+nD] = baxter.compute_control(prediction, q_des[i-1+nD], q_desdot[i-1+nD], q_desddot[i-1+nD])
            if predictions_arr is not None:
                predictors[i] = predictions_arr
        else:
            controls[i-1+nD] = baxter.compute_control(q_vals[i-1], q_des[i-1+nD], q_desdot[i-1+nD], q_desddot[i-1+nD])
            predictors[i] = np.tile(q_vals[i-1], nD).reshape(nD, len(q_vals[i-1]))
        #controls[i-1+nD] = baxter.compute_control(prediction+ np.random.uniform(-0.5, 0.5, 2*dof), q_des[i-1+nD], q_desdot[i-1+nD], q_desddot[i-1+nD])
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

def gen_dataset_dagger(num_data, baxter, dt, T, dof, D, scaling, period, joint_lim_min, joint_lim_max, deviation, predictor_func, model):
    t = np.arange(0, T, dt)
    nD = int(round(D/dt))
    q_des, qd_des, qdd_des = generate_trajectory(joint_lim_min, joint_lim_max, scaling, period, dt, T, D,(joint_lim_max + joint_lim_min) / 2.0  )
    inputs = np.zeros((num_data, nD, 3*dof))
    outputs = np.zeros((num_data, nD, 2*dof))
    index = 0
    num_trajs = 0
    start_time = time.time()
    sample_rate  =  len(np.arange(0, len(t)-1)[nD+1:])
    print("Generating", num_data, "values with sample rate", sample_rate)
    print("This will require simulating", math.ceil(num_data/sample_rate), "trajectories")
    start_time = time.time()
    last_time = start_time
    while index+1 < num_data:
        if num_trajs % 5 == 0:
            print("Simulated Trajectories: ", num_trajs, "/",  math.ceil(num_data/sample_rate))
            print("Total time per this batch of trajectories", time.time()-last_time)
            print("Total time spent", time.time()-start_time)
            last_time = time.time()
        num_trajs += 1
        sample_locs = np.arange(0, len(t)-1)[nD+1:]
        init_cond = init_cond = (joint_lim_max + joint_lim_min) / 2.0  + np.random.uniform(-1*deviation, 1*deviation, dof)
        init_cond = np.array([init_cond, np.zeros(dof)]).reshape(2*dof) 
        states, controls, predictors = simulate_system(baxter, init_cond, q_des, qd_des, qdd_des, dt, T, D, dof, predictor_func, model, randomize=False)
        for i in range(len(sample_locs)):
            val = sample_locs[i]
            myStates = states[val-1] 
            myControls = controls[val-1:val-1+nD] 
            inputs[index, :, 0:2*dof] =  np.tile(myStates, nD).reshape(nD, 2*dof)
            inputs[index, :, 2*dof:] = myControls
            outputs[index] = baxter.compute_predictors(dt, myStates, myControls, None)[1]
            index += 1
            if index+1 >= num_data:
                break
    end_time = time.time()
    inputs = inputs.reshape((num_data, nD*3*dof)).astype(np.float32)
    outputs = outputs.reshape((num_data, nD*2*dof)).astype(np.float32)
    return inputs, outputs

    
def gen_dataset(num_data,baxter, dt, T, dof, D, scaling, period, joint_lim_min, joint_lim_max, deviation, filename):
    t = np.arange(0, T, dt)
    nD = int(round(D/dt))
    q_des, qd_des, qdd_des = generate_trajectory(joint_lim_min, joint_lim_max, scaling, period, dt, T, D,(joint_lim_max + joint_lim_min) / 2.0  )
    inputs = np.zeros((num_data, nD, 3*dof))
    outputs = np.zeros((num_data, nD, 2*dof))
    index = 0
    num_trajs = 0
    start_time = time.time()
    sample_rate  =  len(np.arange(0, len(t)-1)[nD+1:])
    print("Generating", num_data, "values with sample rate", sample_rate)
    print("This will require simulating", math.ceil(num_data/sample_rate), "trajectories")
    start_time = time.time()
    last_time = start_time
    while index+1 < num_data:
    	# random sampling or sample every step. Currently sample every step. 
        # sample_locs = np.random.choice(np.arange(int(len(t)/sample_rate)-1, len(t)-1)[nD+1:], size=sample_rate, replace=False)
        sample_locs = np.arange(0, len(t)-1)[nD+1:]
        if num_trajs % 5 == 0:
            print("Simulated Trajectories: ", num_trajs, "/",  math.ceil(num_data/sample_rate))
            print("Total time per this batch of trajectories", time.time()-last_time)
            print("Total time spent", time.time()-start_time)
            last_time = time.time()
        num_trajs += 1
        init_cond = init_cond = (joint_lim_max + joint_lim_min) / 2.0 
        init_cond = np.array([init_cond, np.zeros(dof)]).reshape(2*dof) 
        states, controls, predictors = simulate_system(baxter, init_cond, q_des, qd_des, qdd_des, dt, T, D, dof, baxter.compute_predictors, None, randomize=True, deviation=deviation)
        for i in range(len(sample_locs)):
            val = sample_locs[i]
            myStates = states[val-1] 
            myControls = controls[val-1:val-1+nD] 
            inputs[index, :, 0:2*dof] =  np.tile(myStates, nD).reshape(nD, 2*dof)
            inputs[index, :, 2*dof:] = myControls
            outputs[index] = baxter.compute_predictors(dt, myStates, myControls, None)[1]
            index += 1
            if index+1 >= num_data:
                break
    end_time = time.time()
    inputs = inputs.reshape((num_data, nD*3*dof))
    outputs = outputs.reshape((num_data, nD*2*dof))
    # Make sure dataset folder is created
    np.save("../datasets/inputs" + filename +".npy", inputs)
    np.save("../datasets/outputs" + filename + ".npy", outputs)
    print("Generated successfully. Total time:", end_time-start_time)
