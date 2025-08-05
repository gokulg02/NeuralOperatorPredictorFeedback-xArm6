import numpy as np
import math

# Creates a Manipulator robot. Implementation is based on Baxter robot
# https://ieeexplore.ieee.org/document/7558740
# "Development of a dynamics model for the Manipulator robot" by Smith Et. Al.

class Manipulator:
    def __init__(self, dof, alpha_mat, beta_mat):
        coms = np.array([[0.00015, 0.02724, -0.01357, 1],\
                 [0.0367, -0.2209, 0.03356, 1],\
                 [0.06977, 0.1135, 0.01163, 1],\
                 [-0.0002, 0.02, -0.026, 1],\
                 [0.06387, 0.0293, 0.0035, 1],\
                 [0, -0.00677, -0.01098, 1]])
        self.coms = coms
        # Ixx, Iyy, Izz, Ixy, Iyz, Ixz
        inertias = np.array([
    [0.005433,   0.004684,    0.0031118,   0.000009864,  -0.000826936, -0.0000268],
    [0.0271342,  0.0053854,   0.0262093,   0.004736,     -0.0047834,    0.00068673],
    [0.006085,   0.0036652,   0.0057045,  -0.0015,        0.0018091,    0.0009558],
    [0.0046981,  0.0042541,   0.00123664, -0.000006486,  -0.0002877,   -0.00001404],
    [0.0013483,  0.00175694,  0.002207,   -0.00042677,    0.0001244,    0.00028758],
    [0.000093,   0.0000587,   0.000132,   -0.0000000,    -0.0000036,   -0.0000000],
])

        # d, a, alpha, m
        dh_params = np.array([[0.267, 0, 0, 2.177],\
        [0, 0, -math.pi/2, 2.011],\
        [0, 0.28949, 0, 1.725],\
        [0.3425, 0.0775, -math.pi, 1.211],\
        [0, 0, math.pi/2, 1.206],\
        [0.097, 0.076, -math.pi/2, 0.17]])
        
        #self.theta_offset = np.array([0, -1.3849, 1.3849, 0, 0, 0])
        self.theta_offset = np.array([0, 0, 0, 0, 0, 0])
        self.theta_offset = self.theta_offset[:dof]

        self.dof = dof
        a = np.zeros(dof)
        d = np.zeros(dof)
        alpha = np.zeros(dof)
        m = np.zeros(dof)
        for i in range(dof):
            a[i] = dh_params[i, 1]
            d[i] = dh_params[i, 0]
            alpha[i] = dh_params[i, 2]
            m[i] = dh_params[i, 3]
        self.a = a
        self.d = d
        self.alpha = alpha
        self.m = m

        J = np.zeros((4,4,dof))
        for i in range(dof):
            J[:,:,i] = [[0.5*(-inertias[i,0]+inertias[i, 1]+inertias[i, 2]),inertias[i, 3],inertias[i,5], m[i]*coms[i, 0]],\
                    [inertias[i, 3], 0.5*(inertias[i,0]-inertias[i, 1]+inertias[i, 2]), inertias[i, 4], m[i]*coms[i, 1]], \
                    [inertias[i, 5], inertias[i, 4], 0.5*(inertias[i,0]+inertias[i, 1]-inertias[i, 2]), m[i]*coms[i, 2]], \
                     [m[i]*coms[i, 0], m[i]*coms[i, 1], m[i]*coms[i, 2], m[i]]]
        self.J = J
            
        Q = np.zeros((4,4))
        Q[0,1] = -1
        Q[1,0] = 1
        self.Q = Q 
        
        self.joint_lim_min = np.array([-6.2832, -2.0595, -3.92699, -6.2832, -1.69297, -6.2832])
        self.joint_lim_min_deriv = -np.ones(6) * np.pi
        self.joint_lim_max = np.array([6.2832, 2.0944, 0.19199, 6.2832, 3.1416, 6.2832])
        self.joint_lim_max_deriv = np.ones(6) * np.pi
        self.joint_lim_min = np.array([self.joint_lim_min[0:self.dof], self.joint_lim_min_deriv[0:self.dof]]).reshape(2*dof)
        self.joint_lim_max = np.array([self.joint_lim_max[0:self.dof], self.joint_lim_max_deriv[0:self.dof]]).reshape(2*dof)
        
        self.min_torques = [-50, -50, -50, -50, -15, -15]
        self.min_torques = self.min_torques[0:self.dof]
        self.max_torques = [50, 50, 50, 50, 15, 15]    
        self.max_torques = self.max_torques[0:self.dof]
        
        self.alpha_mat = alpha_mat
        self.beta_mat = beta_mat

    def modified_dh_transform(self, alpha, a, theta, d):
        ca, sa = math.cos(alpha), math.sin(alpha)
        ct, st = math.cos(theta), math.sin(theta)
        return np.array([
            [ct, -st, 0, a],
            [st * ca, ct * ca, -sa, -d * sa],
            [st * sa, ct * sa, ca, d * ca],
            [0, 0, 0, 1]
        ])


    # computes matrices for sGingle link manipulator
    def matrices_dof_1(self, q, qdot):
        q = q + self.theta_offset
        T01 = self.modified_dh_transform(self.alpha[0], self.a[0], q[0], self.d[0])   
                
        T00 = np.identity(4)
        T11 = np.identity(4)

        return T00, T01, T11

    def matrices_dof_2(self, q, qdot):
        q = q + self.theta_offset
        T01 = self.modified_dh_transform(self.alpha[0], self.a[0], q[0], self.d[0])
        T12 = self.modified_dh_transform(self.alpha[1], self.a[1], q[1], self.d[1])


        T00 = np.identity(4)
        T11 = np.identity(4)
        T22 = np.identity(4)

        T02 = (T01@T12)
        return T00, T01, T02, T11, T12, T22

    def matrices_dof_3(self, q, qdot):
        q = q + self.theta_offset
        T01 = self.modified_dh_transform(self.alpha[0], self.a[0], q[0], self.d[0])                   
        T12 = self.modified_dh_transform(self.alpha[1], self.a[1], q[1], self.d[1])
        T23 = self.modified_dh_transform(self.alpha[2], self.a[2], q[2], self.d[2])


        T00 = np.identity(4)
        T11 = np.identity(4)
        T22 = np.identity(4)
        T33 = np.identity(4)

        T02 = (T01@T12)
        T03 = (T02@T23)

        T13 = (T12@T23)

        return T00, T01, T02, T03, T11, T12, T13, T22, T23, T33


    def matrices_dof_4(self, q, qdot):
        q = q + self.theta_offset
        T01 = self.modified_dh_transform(self.alpha[0], self.a[0], q[0], self.d[0])
        T12 = self.modified_dh_transform(self.alpha[1], self.a[1], q[1], self.d[1])
        T23 = self.modified_dh_transform(self.alpha[2], self.a[2], q[2], self.d[2])         
        T34 = self.modified_dh_transform(self.alpha[3], self.a[3], q[3], self.d[3])
        
        T00 = np.identity(4)
        T11 = np.identity(4)
        T22 = np.identity(4)
        T33 = np.identity(4)
        T44 = np.identity(4)

        T02 = (T01@T12)
        T03 = (T02@T23)
        T04 = (T03@T34)

        T13 = (T12@T23)
        T14 = (T13@T34)

        T24 = (T23@T34)


        return T00, T01, T02, T03, T04, T11, T12, T13, T14, T22, T23, T24, T33, T34, T44

    def matrices_dof_5(self, q, qdot):
        q = q + self.theta_offset
        T01 = self.modified_dh_transform(self.alpha[0], self.a[0], q[0], self.d[0])   
        T12 = self.modified_dh_transform(self.alpha[1], self.a[1], q[1], self.d[1])
        T23 = self.modified_dh_transform(self.alpha[2], self.a[2], q[2], self.d[2])       
        T34 = self.modified_dh_transform(self.alpha[3], self.a[3], q[3], self.d[3])                  
        T45 = self.modified_dh_transform(self.alpha[4], self.a[4], q[4], self.d[4])



        T00 = np.identity(4)
        T11 = np.identity(4)
        T22 = np.identity(4)
        T33 = np.identity(4)
        T44 = np.identity(4)
        T55 = np.identity(4)

        T02 = (T01@T12)
        T03 = (T02@T23)
        T04 = (T03@T34)
        T05 = (T04@T45)

        T13 = (T12@T23)
        T14 = (T13@T34)
        T15 = (T14@T45)
                        
        T24 = (T23@T34)
        T25 = (T24@T45)

        T35 = (T34@T45)

        return T00, T01, T02, T03, T04, T05, T11, T12, T13, T14, T15, T22, T23, T24, T25, T33, T34, T35, T44, T45, T55

    def matrices_dof_6(self, q, qdot):
        q = q + self.theta_offset
        T01 = self.modified_dh_transform(self.alpha[0], self.a[0], q[0], self.d[0])
        T12 = self.modified_dh_transform(self.alpha[1], self.a[1], q[1], self.d[1])
        T23 = self.modified_dh_transform(self.alpha[2], self.a[2], q[2], self.d[2])
        T34 = self.modified_dh_transform(self.alpha[3], self.a[3], q[3], self.d[3])
        T45 = self.modified_dh_transform(self.alpha[4], self.a[4], q[4], self.d[4])
        T56 = self.modified_dh_transform(self.alpha[5], self.a[5], q[5], self.d[5])

        # Identity transforms
        T00 = np.identity(4)
        T11 = np.identity(4)
        T22 = np.identity(4)
        T33 = np.identity(4)
        T44 = np.identity(4)
        T55 = np.identity(4)
        T66 = np.identity(4)

        # Cumulative transforms
        T02 = T01 @ T12
        T03 = T02 @ T23
        T04 = T03 @ T34
        T05 = T04 @ T45
        T06 = T05 @ T56

        T13 = T12 @ T23
        T14 = T13 @ T34
        T15 = T14 @ T45
        T16 = T15 @ T56

        T24 = T23 @ T34
        T25 = T24 @ T45
        T26 = T25 @ T56

        T35 = T34 @ T45
        T36 = T35 @ T56

        T46 = T45 @ T56

        return T00, T01, T02, T03, T04, T05, T06, T11, T12, T13, T14, T15, T16, T22, T23, T24, T25, T26, T33, T34, T35, T36, T44, T45, T46, T55, T56, T66

    def compute_matrices(self, q, qdot):
        dof = self.dof
        match self.dof:
            case 1:
                T00, T01, T11 = self.matrices_dof_1(q, qdot)
            case 2:
                T00, T01, T02, T11, T12, T22 = self.matrices_dof_2(q, qdot)
            case 3:
                T00, T01, T02, T03, T11, T12, T13, T22, T23, T33 = self.matrices_dof_3(q, qdot)
            case 4:
                T00, T01, T02, T03, T04, T11, T12, T13, T14, T22, T23, T24,  T33, T34, T44 = self.matrices_dof_4(q, qdot)
            case 5:
                T00, T01, T02, T03, T04, T05, T11, T12, T13, T14, T15, T22, T23, T24, T25, T33, T34, T35, T44, T45, T55 = self.matrices_dof_5(q, qdot)
            case 6:
                T00, T01, T02, T03, T04, T05, T06, T11, T12, T13, T14, T15, T16, T22, T23, T24, T25, T26, T33, T34, T35, T36, T44, T45, T46, T55, T56, T66 = self.matrices_dof_6(q, qdot)
            case _:
                raise Exception("DOF not supported. Please choose 1<=dof<=5.")

        U = np.zeros((4,4,dof,dof));
        for i in range(1,dof+1):
            for j in range(1, dof+1):
                if j <= i:
                    U[:,:,i-1,j-1] = eval('T'+str(0)+str(j-1)) @ self.Q @ eval('T' + str(j-1) + str(i))
                else:
                    U[:,:,i-1,j-1] = np.zeros((4, 4)); 
        
        U3 = np.zeros((4,4,dof, dof, dof));
        for i in range(1, dof+1):
            for j in range(1, dof+1):
                for k in range(1, dof+1):
                    if (i >= k) and (k >= j):
                        U3[:,:,i-1,j-1,k-1] = eval('T'+str(0)+str(j-1)) @ self.Q @ eval('T'+str(j-1)+str(k-1)) @ self.Q @ eval('T'+str(k-1)+str(i))
                    elif (i >= j) and (j >= k):
                        U3[:,:,i-1,j-1,k-1] = eval('T'+str(0)+str(k-1)) @ self.Q @ eval('T'+str(k-1)+str(j-1)) @ self.Q @ eval('T'+str(j-1)+str(i))
                    else:
                        U3[:,:,i-1,j-1,k-1] = np.zeros((4, 4))

        
        # Inertia
        D = np.zeros((dof, dof))
        for i in range(0, dof):
            for k in range(0, dof):
                res = 0;
                for j in range(max(i,k), dof):
                    res = res + np.trace(U[:,:,j,k]@self.J[:,:,j]@(U[:,:,j,i].T))
                D[i,k] = res;

        # Coriolis
        h3 = np.zeros((dof, dof, dof))
        for i in range(0, dof):
            for k in range(0, dof):
                for m_idx in range(0, dof):
                    res = 0;
                    for j in range(max([i, k, m_idx]), dof):
                        res = res + np.trace(U3[:,:,j,k,m_idx] @ self.J[:,:,j] @ (U[:,:,j,i].T))
                    h3[i,k,m_idx] = res;
        h = np.zeros(dof);
        for i in range(dof):
            sum_k = 0
            for k in range(dof):
                sum_m = 0
                for m_idx in range(dof):
                    sum_m = sum_m + h3[i,k,m_idx] * qdot[k] * qdot[m_idx]
                sum_k = sum_k + sum_m
            h[i] = sum_k;

        # Gavity
        g = 9.81
        g_vec = np.array([0, 0, -g, 0]);
        c = np.zeros(dof);
        for i in range(dof):
            res = 0;
            for j in range(dof):
                res = res + (-self.m[j] * g_vec @ U[:,:,j,i] @ self.coms[j]);
            c[i] = res;

        return D, h, c

    def step_q(self, states, torque):
        x1 = states[:self.dof]
        x2 = states[self.dof:]
        M, C, G = self.compute_matrices(x1, x2)
        x1dot = x2
        x2dot = np.linalg.inv(M) @ (torque - C @ x2 - G)
        return np.array([x1dot, x2dot]).reshape(2*self.dof)

    def get_error_states(self, states, qdes, qdesdot):
        x1 = states[:self.dof]
        x2 = states[self.dof:]
        e1 = qdes - x1
        e2 = (qdesdot - x2) + self.alpha_mat @ e1
        return np.array([e1, e2]).reshape(2*self.dof)

    def compute_control(self, states, qdes, qdesdot, qdesddot):
        x1 = states[:self.dof]
        x2 = states[self.dof:]
        error_states = self.get_error_states(states, qdes, qdesdot)
        e1 = error_states[:self.dof]
        e2 = error_states[self.dof:]
        M, C, G = self.compute_matrices(x1, x2)
        h = qdesddot - self.alpha_mat @ self.alpha_mat @ e1 + np.linalg.inv(M) @ (C@qdesdot + G + C @ self.alpha_mat @ e1 - C @ e2)
        return self.saturate_torques(M @ (h + (self.beta_mat + self.alpha_mat)@ e2))

    def saturate_torques(self, torque):
        return np.maximum(self.min_torques, np.minimum(self.max_torques, torque))

    def saturate_joints(self, joints):
        return np.maximum(self.joint_lim_min, np.minimum(self.joint_lim_max, joints))

    def compute_predictors(self, dt, X, controls, _model):
        res = np.zeros((len(controls)+1, len(X)))
        res[0] = np.copy(X)
        for i in range(len(controls)):
            res[i+1] = self.saturate_joints(res[i] + dt*self.step_q(res[i], controls[i]))
        return res[-1], res[1:]
