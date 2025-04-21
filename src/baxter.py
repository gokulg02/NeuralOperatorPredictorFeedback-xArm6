import numpy as np
import math

# Creates a Baxter robot. Implementation is based on 
# https://ieeexplore.ieee.org/document/7558740
# "Development of a dynamics model for the Baxter robot" by Smith Et. Al.

class Baxter:
    def __init__(self, dof, alpha_mat, beta_mat):
        coms = np.array([[-0.05117, 0.07908, 0.00086, 1],\
                 [0.00269, -0.00529, 0.06845, 1],\
                 [-0.07176, 0.08149, 0.00132, 1],\
                 [0.00159, -0.01117, 0.02618, 1],\
                 [-0.01168, 0.13111, 0.0046, 1],\
                 [0.00697, 0.006, 0.06048, 1],\
                 [0.005137, 0.0009572, -0.06682, 1]])
        self.coms = coms
        # Ixx, Iyy, Izz, Ixy, Iyz, Ixz
        inertias = np.array([[0.0470910226, 0.035959884, 0.0376697645, -0.0061487003, -0.0007808689, 0.0001278755],\
        [0.027885975, 0.020787492, 0.0117520941,-0.0001882199, 0.0020767576, -0.00030096397],\
        [0.0266173355, 0.012480083, 0.0284435520,-0.0039218988, -0.001083893, 0.0002927063],\
        [0.0131822787, 0.009268520, 0.0071158268,-0.0001966341, 0.000745949, 0.0003603617],\
        [0.0166774282, 0.003746311, 0.0167545726,-0.0001865762, 0.0006473235, 0.0001840370],\
        [0.0070053791, 0.005527552, 0.0038760715,0.0001534806, -0.0002111503, -0.000443847],\
        [0.0008162135, 0.0008735012, 0.0005494148,0.000128440, 0.0001057726, 0.00018969891]])

        # theta, d, a, alpha, m
        dh_params = np.array([[0.2703, 0.069, -math.pi/2, 5.70044],\
        [0, 0, math.pi/2, 3.22698],\
        [0.3644, 0.069, -math.pi/2, 4.31272],\
        [0, 0, math.pi/2, 2.07206],\
        [0.3743, 0.01, -math.pi/2, 2.24665],\
        [0, 0, math.pi/2, 1.60979],\
        [0.2295,  0, 0, 0.54218]])

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
        
        self.joint_lim_min = np.array([-1.7016, -2.147, -3.0541, -0.05, -3.059, -1.571, -3.059])
        self.joint_lim_min_deriv = np.array([-2, -2, -2, -2, -4, -4, -4])
        self.joint_lim_max = np.array([1.7016, 1.047, 3.0541, 2.618, 3.059, 2.094, 3.059])
        self.joint_lim_max_deriv = np.array([2, 2, 2, 2, 4, 4, 4])
        self.joint_lim_min = np.array([self.joint_lim_min[0:self.dof], self.joint_lim_min_deriv[0:self.dof]]).reshape(2*dof)
        self.joint_lim_max = np.array([self.joint_lim_max[0:self.dof], self.joint_lim_max_deriv[0:self.dof]]).reshape(2*dof)
        
        self.min_torques = [-50, -50, -50, -50, -15, -15, -15]
        self.min_torques = self.min_torques[0:self.dof]
        self.max_torques = [50, 50, 50, 50, 15, 15, 15]    
        self.max_torques = self.max_torques[0:self.dof]
        
        self.alpha_mat = alpha_mat
        self.beta_mat = beta_mat

    # computes matrices for sGingle link manipulator
    def matrices_dof_1(self, q, qdot):
        T01 = np.array([[math.cos(q[0]),  -math.cos(self.alpha[0])*math.sin(q[0]),  math.sin(self.alpha[0])*math.sin(q[0]), self.a[0]*math.cos(q[0])],\
            [math.sin(q[0]),   math.cos(self.alpha[0])*math.cos(q[0]), -math.sin(self.alpha[0])*math.cos(q[0]), self.a[0]*math.sin(q[0])],\
            [0,           math.sin(self.alpha[0]),            math.cos(self.alpha[0]),           self.d[0]],\
            [0,           0,                        0,                       1]])   

                
        T00 = np.identity(4)
        T11 = np.identity(4)

        return T00, T01, T11

    def matrices_dof_2(self, q, qdot):
        T01 = np.array([[math.cos(q[0]),  -math.cos(self.alpha[0])*math.sin(q[0]),  math.sin(self.alpha[0])*math.sin(q[0]), self.a[0]*math.cos(q[0])],\
            [math.sin(q[0]),   math.cos(self.alpha[0])*math.cos(q[0]), -math.sin(self.alpha[0])*math.cos(q[0]), self.a[0]*math.sin(q[0])],\
            [0,           math.sin(self.alpha[0]),            math.cos(self.alpha[0]),           self.d[0]],\
            [0,           0,                        0,                       1]])   

                
        T12 = np.array([[math.cos(q[1]),  -math.cos(self.alpha[1])*math.sin(q[1]),  math.sin(self.alpha[1])*math.sin(q[1]), self.a[1]*math.cos(q[1])],\
            [math.sin(q[1]),   math.cos(self.alpha[1])*math.cos(q[1]), -math.sin(self.alpha[1])*math.cos(q[1]), self.a[1]*math.sin(q[1])],\
            [0,           math.sin(self.alpha[1]),            math.cos(self.alpha[1]),           self.d[1]],\
            [0,           0,                        0,                       1]])


        T00 = np.identity(4)
        T11 = np.identity(4)
        T22 = np.identity(4)

        T02 = (T01@T12)
        return T00, T01, T02, T11, T12, T22

    def matrices_dof_3(self, q, qdot):
        T01 = np.array([[math.cos(q[0]),  -math.cos(self.alpha[0])*math.sin(q[0]),  math.sin(self.alpha[0])*math.sin(q[0]), self.a[0]*math.cos(q[0])],\
            [math.sin(q[0]),   math.cos(self.alpha[0])*math.cos(q[0]), -math.sin(self.alpha[0])*math.cos(q[0]), self.a[0]*math.sin(q[0])],\
            [0,           math.sin(self.alpha[0]),            math.cos(self.alpha[0]),           self.d[0]],\
            [0,           0,                        0,                       1]])   

                
        T12 = np.array([[math.cos(q[1]),  -math.cos(self.alpha[1])*math.sin(q[1]),  math.sin(self.alpha[1])*math.sin(q[1]), self.a[1]*math.cos(q[1])],\
            [math.sin(q[1]),   math.cos(self.alpha[1])*math.cos(q[1]), -math.sin(self.alpha[1])*math.cos(q[1]), self.a[1]*math.sin(q[1])],\
            [0,           math.sin(self.alpha[1]),            math.cos(self.alpha[1]),           self.d[1]],\
            [0,           0,                        0,                       1]])

        T23 = np.array([[math.cos(q[2]),  -math.cos(self.alpha[2])*math.sin(q[2]),  math.sin(self.alpha[2])*math.sin(q[2]), self.a[2]*math.cos(q[2])],\
            [math.sin(q[2]),   math.cos(self.alpha[2])*math.cos(q[2]), -math.sin(self.alpha[2])*math.cos(q[2]), self.a[2]*math.sin(q[2])],\
            [0,           math.sin(self.alpha[2]),            math.cos(self.alpha[2]),           self.d[2]],\
            [0,           0,                        0,                       1]])


        T00 = np.identity(4)
        T11 = np.identity(4)
        T22 = np.identity(4)
        T33 = np.identity(4)

        T02 = (T01@T12)
        T03 = (T02@T23)

        T13 = (T12@T23)

        return T00, T01, T02, T03, T11, T12, T13, T22, T23, T33


    def matrices_dof_4(self, q, qdot):
        T01 = np.array([[math.cos(q[0]),  -math.cos(self.alpha[0])*math.sin(q[0]),  math.sin(self.alpha[0])*math.sin(q[0]), self.a[0]*math.cos(q[0])],\
            [math.sin(q[0]),   math.cos(self.alpha[0])*math.cos(q[0]), -math.sin(self.alpha[0])*math.cos(q[0]), self.a[0]*math.sin(q[0])],\
            [0,           math.sin(self.alpha[0]),            math.cos(self.alpha[0]),           self.d[0]],\
            [0,           0,                        0,                       1]])   

                
        T12 = np.array([[math.cos(q[1]),  -math.cos(self.alpha[1])*math.sin(q[1]),  math.sin(self.alpha[1])*math.sin(q[1]), self.a[1]*math.cos(q[1])],\
            [math.sin(q[1]),   math.cos(self.alpha[1])*math.cos(q[1]), -math.sin(self.alpha[1])*math.cos(q[1]), self.a[1]*math.sin(q[1])],\
            [0,           math.sin(self.alpha[1]),            math.cos(self.alpha[1]),           self.d[1]],\
            [0,           0,                        0,                       1]])

        T23 = np.array([[math.cos(q[2]),  -math.cos(self.alpha[2])*math.sin(q[2]),  math.sin(self.alpha[2])*math.sin(q[2]), self.a[2]*math.cos(q[2])],\
            [math.sin(q[2]),   math.cos(self.alpha[2])*math.cos(q[2]), -math.sin(self.alpha[2])*math.cos(q[2]), self.a[2]*math.sin(q[2])],\
            [0,           math.sin(self.alpha[2]),            math.cos(self.alpha[2]),           self.d[2]],\
            [0,           0,                        0,                       1]])

                        
        T34 = np.array([[math.cos(q[3]),  -math.cos(self.alpha[3])*math.sin(q[3]),  math.sin(self.alpha[3])*math.sin(q[3]), self.a[3]*math.cos(q[3])],\
            [math.sin(q[3]),     math.cos(self.alpha[3])*math.cos(q[3]), -math.sin(self.alpha[3])*math.cos(q[3]), self.a[3]*math.sin(q[3])],\
            [0,                 math.sin(self.alpha[3]),                 math.cos(self.alpha[3]),               self.d[3]],\
            [0,           0,                        0,                       1]])

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
        T01 = np.array([[math.cos(q[0]),  -math.cos(self.alpha[0])*math.sin(q[0]),  math.sin(self.alpha[0])*math.sin(q[0]), self.a[0]*math.cos(q[0])],\
            [math.sin(q[0]),   math.cos(self.alpha[0])*math.cos(q[0]), -math.sin(self.alpha[0])*math.cos(q[0]), self.a[0]*math.sin(q[0])],\
            [0,           math.sin(self.alpha[0]),            math.cos(self.alpha[0]),           self.d[0]],\
            [0,           0,                        0,                       1]])   

                
        T12 = np.array([[math.cos(q[1]),  -math.cos(self.alpha[1])*math.sin(q[1]),  math.sin(self.alpha[1])*math.sin(q[1]), self.a[1]*math.cos(q[1])],\
            [math.sin(q[1]),   math.cos(self.alpha[1])*math.cos(q[1]), -math.sin(self.alpha[1])*math.cos(q[1]), self.a[1]*math.sin(q[1])],\
            [0,           math.sin(self.alpha[1]),            math.cos(self.alpha[1]),           self.d[1]],\
            [0,           0,                        0,                       1]])

        T23 = np.array([[math.cos(q[2]),  -math.cos(self.alpha[2])*math.sin(q[2]),  math.sin(self.alpha[2])*math.sin(q[2]), self.a[2]*math.cos(q[2])],\
            [math.sin(q[2]),   math.cos(self.alpha[2])*math.cos(q[2]), -math.sin(self.alpha[2])*math.cos(q[2]), self.a[2]*math.sin(q[2])],\
            [0,           math.sin(self.alpha[2]),            math.cos(self.alpha[2]),           self.d[2]],\
            [0,           0,                        0,                       1]])

                        
        T34 = np.array([[math.cos(q[3]),  -math.cos(self.alpha[3])*math.sin(q[3]),  math.sin(self.alpha[3])*math.sin(q[3]), self.a[3]*math.cos(q[3])],\
            [math.sin(q[3]),     math.cos(self.alpha[3])*math.cos(q[3]), -math.sin(self.alpha[3])*math.cos(q[3]), self.a[3]*math.sin(q[3])],\
            [0,                 math.sin(self.alpha[3]),                 math.cos(self.alpha[3]),               self.d[3]],\
            [0,           0,                        0,                       1]])

                                    
        T45 = np.array([[math.cos(q[4]),  -math.cos(self.alpha[4])*math.sin(q[4]),  math.sin(self.alpha[4])*math.sin(q[4]), self.a[4]*math.cos(q[4])],\
            [math.sin(q[4]),     math.cos(self.alpha[4])*math.cos(q[4]), -math.sin(self.alpha[4])*math.cos(q[4]), self.a[4]*math.sin(q[4])],\
            [0,                 math.sin(self.alpha[4]),                 math.cos(self.alpha[4]),               self.d[4]],\
            [0,           0,                        0,                       1]])





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
