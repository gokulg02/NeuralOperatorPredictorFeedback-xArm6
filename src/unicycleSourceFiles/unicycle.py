import numpy as np
import math

def dynamics(state, control):
    return np.array([control[0]*math.cos(state[2]), control[0]*math.sin(state[2]), control[1]])

def controller(state, t):
    p = state[0]*math.cos(state[2])+state[1]*math.sin(state[2])
    q = state[0]*math.sin(state[2])-state[1]*math.cos(state[2])
    omega = -5*p**2*math.cos(3*t) - p*q*(1+25*math.cos(3*t)*math.cos(3*t))-state[2]
    nu = -p+5*q*math.sin(3*t)-math.cos(3*t) + q*omega
    return np.array([nu, omega])

def predictor_const_delay(cur_state, control_history, D, x, _model=None):
    nx = len(x)
    integral_tmp = np.zeros((nx, 3))
    res = np.zeros((nx, 3))
    res[0] = np.copy(cur_state)
    prev = res.copy()
    for i in range(1, nx):
        integral_tmp[i] = dynamics(res[i-1], control_history[i-1]) 
        res[i] = res[0] + D*np.array([np.trapz(integral_tmp[0:i+1, 0], x=x[0:i+1]), \
                                      np.trapz(integral_tmp[0:i+1, 1], x=x[0:i+1]),\
                                      np.trapz(integral_tmp[0:i+1, 2], x=x[0:i+1])])
    return res

def predictor_non_const_delay(cur_state, control_history, D, x, _model=None):
    nx = len(x)
    integral_tmp = np.zeros((nx, 3))
    res = np.zeros((nx, 3))
    res[0] = np.copy(cur_state)
    for i in range(1, nx):
        integral_tmp[i] = dynamics(res[i-1], control_history[i-1]) 
        res[i] = res[0] + D*np.array([np.trapz(integral_tmp[0:i+1, 0], x=x[0:i+1]), \
                                      np.trapz(integral_tmp[0:i+1, 1], x=x[0:i+1]),\
                                      np.trapz(integral_tmp[0:i+1, 2], x=x[0:i+1])])
    return res[-1]


def simulate_system_no_delay(init_cond, dt, T):
    t = np.arange(0, T, dt)
    u = np.zeros((len(t), 3))
    u[0] = init_cond
    controls = np.zeros((len(t), 2))
    for i in range(1, len(t)):
        control = controller(u[i-1], t[i-1])
        u[i] = u[i-1] + dt*dynamics(u[i-1], control)
        controls[i] = control
    return u, controls
    
def simulate_system_const_delay(init_cond, dt, T, dx, D, predictor_func=None, model=None):
    t = np.arange(0, T, dt)
    x = np.arange(0, 1, dx)
    nx = len(x)
    nt = len(t)
    u = np.zeros((nt, 3))
    pde_sol = np.zeros((nt, nx, 2))
    controls = np.zeros((nt, 2))
    predictions = np.zeros((nt, nx, 3))
    if predictor_func is None:
        predictor_func = predictor_const_delay
    u[0] = init_cond
    for i in range(1, nt):
        prediction = predictor_func(u[i-1], pde_sol[i-1], D, x, model)
        predictions[i-1] = prediction
        pde_sol[i][0:nx-1] = pde_sol[i-1][0:nx-1] + dt/(D*dx) * (pde_sol[i-1][1:nx] - pde_sol[i-1][0:nx-1])
        pde_sol[i][-1] = controller(prediction[-1], t[i-1]+D)
        u[i] = u[i-1] + dt*dynamics(u[i-1], pde_sol[i][0])
        controls[i] = pde_sol[i][0]
    return u, controls, pde_sol, predictions
    
def simulate_system_non_const_delay(init_cond, dt, T, dx, delay_funcs, args, predictor_func=None, model=None):
    phi_inv, phi_inv_deriv = delay_funcs[0], delay_funcs[1]
    t = np.arange(0, T, dt)
    x = np.arange(0, 1, dx)
    nx = len(x)
    nt = len(t)
    u = np.zeros((nt, 3))
    pde_sol = np.zeros((nt, nx, 2))
    controls = np.zeros((nt, 2))
    predictions = np.zeros((nt, nx, 3))
    delays = np.zeros((nt))
    u[0] = init_cond
    for i in range(1, nt):
        prediction = predictor_non_const_delay(u[i-1], pde_sol[i-1], phi_inv(t[i-1], args[0], args[1])-t[i-1], x, model)
        predictions[i-1] = prediction
        delays[i-1] = phi_inv(t[i-1], args[0], args[1])-t[i-1]
        pi = (1+x[0:nx-1]*(phi_inv_deriv(t[i-1], args[0], args[1])-1)/(phi_inv(t[i-1], args[0], args[1])-t[i-1]))
        pde_sol[i][0:nx-1] = pde_sol[i-1][0:nx-1] + dt/dx * np.tile(pi[:, np.newaxis], (1, 2))\
                                                             *(pde_sol[i-1][1:nx] - pde_sol[i-1][0:nx-1])
        pde_sol[i][-1] = controller(prediction[-1], t[i-1]+D)
        u[i] = u[i-1] + dt*dynamics(u[i-1], pde_sol[i][0])
        controls[i] =  pde_sol[i][0]
    return u, controls, pde_sol, predictions, delays





