"""
A Simple Pendulum Gate

# ----------------------
# p = pivot point
# c = center
# 1,2,3,4 = corners
# ----------------------
#           p           
#           |
#           |
#           |
#           |
#           |
#           |
#   2 - - - - - - - 1 
#   |               |
#   |       c       |
#   |               |
#   4 - - - - - - - 3 
#
"""

import numpy as np
from high_mpc.common.quad_index import *
from high_mpc.common.pend_index import *
from high_mpc.common.util import Point

#
class Pendulum_v1(object):
    #
    def __init__(self, pivot_point, sigma, T, dt):
        self.s_dim = 2
        self.a_dim = 0
        self._length = 2.0   
        self._damping = 0.1
        self._mass = 2.0
        self._pi = 3.141592
        self._gz = 9.81
        self._dt = dt
        self.pivot_point = pivot_point # e.g., np.array([2.0, 0.0, 2.0])
        self._T = T
        #
        self.sigma = sigma
        self._N = int(T/dt)
        self.width = 2.0
        self.height = 1
    
    def plan(self, state, opt_t=1.0):
        #
        plans, pred_traj = [], []
        M = 4
        DT = self._dt/M
        #
        for i in range(self._N):
            X = state
            for _ in range(M):
                k1 = DT * self._f(X)
                k2 = DT * self._f(X + 0.5 * k1)
                k3 = DT * self._f(X + 0.5 * k2)
                k4 = DT * self._f(X + k3)
                #
                X = X + (k1 + 2.0*(k2 + k3) + k4)/6.0
            #
            state = X
            traj_euler_point = self.get_cartesian_state(state, euler=True).tolist()
            
            # plan trajectory and optimal time & optimal vx
            traj_quat_point = self.get_cartesian_state(state, euler=False).tolist()
            # traj_quat_point[kPosX] = opt_vx
            
            current_t = i * self._dt
            plan_i = traj_quat_point + [current_t, opt_t, self.sigma]
    
            #
            plans += plan_i
            pred_traj.append(traj_euler_point)
        
        return plans, pred_traj

    def _f(self, state):
        #
        theta = state[0]
        dot_theta = state[1]
        return np.array([dot_theta, \
            -((self._gz/self._length)*np.sin(theta)+(self._damping/self._mass)*dot_theta)])
        
    def get_cartesian_state(self, state, euler=True):
        if not euler:
            cstate = np.zeros(shape=10)
            cstate[kPosX:kPosZ+1] = self.get_position(state)
            cstate[kQuatW:kQuatZ+1] = self.get_quaternion(state)
            cstate[kVelX:kVelZ+1] = self.get_veloctiy(state)
            return cstate
        else:
            cstate = np.zeros(shape=9)
            cstate[0:3] = self.get_position(state)
            cstate[3:6] = self.get_euler(state)
            cstate[6:9] = self.get_veloctiy(state)
            return cstate

    def get_position(self, state):
        pos = np.zeros(shape=3)
        pos[0] = self.pivot_point[0]
        pos[1] = self.pivot_point[1] + self._length*np.sin(state[kTheta])
        pos[2] = self.pivot_point[2] - self._length*np.cos(state[kTheta])
        return pos

    def get_veloctiy(self, state):
        vel = np.zeros(shape=3)
        vel[0] = 0.0
        vel[1] = self._length*state[kDotTheta]*np.cos(state[kTheta])
        vel[2] = self._length*state[kDotTheta]*np.sin(state[kTheta])
        return vel

    def get_euler(self, state):
        euler = np.zeros(shape=3)
        euler[0] = state[kTheta]
        euler[1] = 0.0 
        euler[2] = 0.0 
        return euler
    
    def get_quaternion(self, state):
        roll, pitch, yaw = self.get_euler(state)
        #
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        #
        qw = cy * cp * cr + sy * sp * sr
        qx = cy * cp * sr - sy * sp * cr
        qy = sy * cp * sr + cy * sp * cr
        qz = sy * cp * cr - cy * sp * sr
        #
        return [qw, qx, qy, qz]

