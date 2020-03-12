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
from high_mpc.common.pend_index import *
from high_mpc.common.util import Point

class Pendulum_v0(object):
    #
    def __init__(self, pivot_point, dt):
        self.s_dim = 2
        self.a_dim = 0
        
        #
        self._damping = 0.1
        self._mass = 2.0
        self._gz = 9.81
        self._dt = dt
        self.pivot_point = pivot_point # e.g., np.array([2.0, 0.0, 2.0])
        
        self._state = np.zeros(shape=self.s_dim)
        
        # initial state
        self._theta_box = np.array([-0.8, 0.8]) * np.pi
        self._dot_theta_box = np.array([-0.1, 0.1]) * np.pi

        # self._theta_box = np.array([0.5, 0.5]) * self._pi
        # self._dot_theta_box = np.array([-0.0, 0.0]) * self._pi

        # x, y, z, roll, pitch, yaw, vx, vy, vz
        self.obs_low = np.array([-10, -10, -10, -np.pi, -np.pi, -np.pi, -10, -10, -10])
        self.obs_high = np.array([10, 10, 10, np.pi, np.pi, np.pi, 10, 10, 10])

        # compute the distance between pivot point and 4 corners
        self.length = 2.0  # distance between pivot point to the gate center
        self.width = 1.0   # gate width (for visualization only)
        self.height = 0.5  # gate heiht (for visualization only)
            
        #
        self._init_corners()
        # self.reset()
        self._t = 0.0
    
    def _init_corners(self):
        # compute distance between pivot point to four corners
        # and the 4 angles (for visualization)
        edge1, edge2 = self.width/2,  self.length-self.height/2
        self.length1 = np.sqrt( (edge1)**2 + (edge2)**2 )
        self.delta_theta1 = np.arctan2(edge1, edge2)
        #
        self.length2 = self.length1
        self.delta_theta2 = -self.delta_theta1

        #
        edge1, edge2 = self.width/2, self.length+self.height/2
        self.length3 = np.sqrt( (edge1)**2 + (edge2)**2 )
        self.delta_theta3 = np.arctan2(edge1, edge2)
        #
        self.length4 = self.length3
        self.delta_theta4 = -self.delta_theta3
        
    def reset(self, init_theta=None):
        if init_theta is not None:
            self._state[kTheta] = init_theta
        else:
            self._state[kTheta] = np.random.uniform( \
                low=self._theta_box[0], high=self._theta_box[1])
        self._state[kDotTheta] = np.random.uniform( \
            low=self._dot_theta_box[0], high=self._dot_theta_box[1])
        #
        self._t = 0.0
        # print("init pendulum: ", self._state)
        return self._state

    def run(self,):
        self._t = self._t + self._dt
        
        # rk4 int
        M = 4
        DT = self._dt/M
        
        X = self._state
        for _ in range(M):
            k1 = DT * self._f(X)
            k2 = DT * self._f(X + 0.5 * k1)
            k3 = DT * self._f(X + 0.5 * k2)
            k4 = DT * self._f(X + k3)
            #
            X = X + (k1 + 2.0*(k2 + k3) + k4)/6.0
        #
        self._state = X
        return self._state

    def _f(self, state):
        #
        theta = state[0]
        dot_theta = state[1]
        return np.array([dot_theta, \
            -((self._gz/self.length)*np.sin(theta)+(self._damping/self._mass)*dot_theta)])

    def get_state(self,):
        return self._state
        
    def get_cartesian_state(self):
        cartesian_state = np.zeros(shape=9)
        cartesian_state[0:3] = self.get_position()
        cartesian_state[3:6] = self.get_euler()
        cartesian_state[6:9] = self.get_veloctiy()
        return cartesian_state
    
    def get_position(self,):
        pos = np.zeros(shape=3)
        pos[0] = self.pivot_point[0]
        pos[1:] = self._to_planar_coordinates(self.pivot_point, \
            l=self.length, theta=self._state[kTheta])
        return pos

    def get_veloctiy(self,):
        vel = np.zeros(shape=3)
        vel[0] = 0.0
        vel[1] = self.length*self._state[kDotTheta]*np.cos(self._state[kTheta])
        vel[2] = self.length*self._state[kDotTheta]*np.sin(self._state[kTheta])
        return vel

    def get_euler(self,):
        euler = np.zeros(shape=3)
        euler[0] = self._state[kTheta]
        euler[1] = 0.0 
        euler[2] = 0.0 
        return euler

    @property
    def t(self):
        return self._t

    @staticmethod
    def _to_planar_coordinates(pivot_point, l, theta):
        y = pivot_point[1] + l*np.sin(theta)
        z = pivot_point[2] - l*np.cos(theta)
        return y, z

    def get_corners(self, ):  
        theta = self._state[kTheta]
        y1, z1 = self._to_planar_coordinates(self.pivot_point, self.length1, theta+self.delta_theta1)
        y2, z2 = self._to_planar_coordinates(self.pivot_point, self.length2, theta+self.delta_theta2)
        y3, z3 = self._to_planar_coordinates(self.pivot_point, self.length3, theta+self.delta_theta3)
        y4, z4 = self._to_planar_coordinates(self.pivot_point, self.length4, theta+self.delta_theta4)
        #
        corners = [ Point(x=y1, y=z1), Point(x=y2, y=z2), Point(x=y3, y=z3), Point(x=y4, y=z4) ]
        return corners

    def get_3d_corners(self,):
        theta = self._state[kTheta]
        y1, z1 = self._to_planar_coordinates(self.pivot_point, self.length1, theta+self.delta_theta1)
        y2, z2 = self._to_planar_coordinates(self.pivot_point, self.length2, theta+self.delta_theta2)
        y3, z3 = self._to_planar_coordinates(self.pivot_point, self.length3, theta+self.delta_theta3)
        y4, z4 = self._to_planar_coordinates(self.pivot_point, self.length4, theta+self.delta_theta4)
        #
        x = self.pivot_point[0]
        corners_3d = [[x, y1, z1], [x, y2, z2 ], [x, y3, z3 ], [x, y4, z4]]
        return corners_3d

if __name__ == "__main__":
    # test run
    import matplotlib.pyplot as plt
    dt = 0.02
    tf = 20.0
    #
    pivot = [2.0, 2.0, 2.0] # x, y, z

    # # # # # # # # # # # # # # # # # # #
    # -- test Pendulum v0
    # # # # # # # # # # # # # # # # # #
    env = Pendulum_v0(pivot, dt=0.02)
    l_t, l_pos, l_vel, l_theta  = [], [], [], []
    #
    env.reset()
    #
    while env.t < tf:
        #
        l_t.append(env.t)
        l_pos.append(env.get_position())
        l_vel.append(env.get_veloctiy())
        l_theta.append(env.get_euler())
        #
        env.run()
    #
    l_pos = np.asarray(l_pos)
    l_vel = np.asarray(l_vel)
    l_theta = np.asarray(l_theta)
    #
    fig, axes = plt.subplots(3, 1) 
    axes[0].plot(l_t, l_pos[:, 0], '-r', label="x")
    axes[0].plot(l_t, l_pos[:, 1], '-g', label="y")
    axes[0].plot(l_t, l_pos[:, 2], '-b', label="z")
    axes[0].legend()
    #
    axes[1].plot(l_t, l_vel[:, 0], '-r', label="vx")
    axes[1].plot(l_t, l_vel[:, 1], '-g', label="vy")
    axes[1].plot(l_t, l_vel[:, 2], '-b', label="vz")
    axes[1].legend()
    #
    axes[2].plot(l_t, l_theta[:, 0], '-r', label="roll")
    axes[2].plot(l_t, l_theta[:, 1], '-g', label="pitch")
    axes[2].plot(l_t, l_theta[:, 2], '-b', label="yaw")
    axes[2].legend()

    #
    plt.show()



        
