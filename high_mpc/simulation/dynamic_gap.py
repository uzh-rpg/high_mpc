import numpy as np
#
from high_mpc.env.quadrotor import Quadrotor_v0
from high_mpc.env.pendulum import Pendulum_v0
#
from high_mpc.nmpc.nmpc_v0 import NMPC_v0
from high_mpc.nmpc.nmpc_v1 import NMPC_v1
from high_mpc.planner.pendulum import Pendulum_v1
#
from high_mpc.common.quad_index import *

#
class Space(object):

    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.shape = self.low.shape

    def sample(self):
        return np.random.uniform(self.low, self.high)

class DynamicGap_v0(object):

    def __init__(self, so_path, sigma=10, version=0):
        # 
        self.goal_point = np.array([4.0,  0.0, 2.0])
        self.pivot_point = np.array([2.0, 0.0, 3.0])

        # goal state, position, quaternion, velocity
        self.quad_sT = self.goal_point.tolist() + [1.0, 0.0, 0.0, 0.0] + [0.0, 0.0, 0.0] 

        # simulation parameters ....
        self.sim_T = 3.0    # Episode length, seconds
        self.sim_dt = 0.02  # simulation time step
        self.max_episode_steps = int(self.sim_T/self.sim_dt)
        # Simulators, a quadrotor and a pendulum
        self.quad = Quadrotor_v0(dt=self.sim_dt)
        self.pend = Pendulum_v0(self.pivot_point, dt=self.sim_dt)

        # Planning parameters, for MPC...
        self.plan_T = 2.0   # Prediction horizon for MPC and local planner
        self.plan_dt = 0.04 # Sampling time step for MPC and local planner
        #
        if version==0:
            self.nmpc = NMPC_v0(T=self.plan_T, dt=self.plan_dt, so_path=so_path)
        else:
            self.nmpc = NMPC_v1(T=self.plan_T, dt=self.plan_dt, so_path=so_path)
        self.planner = Pendulum_v1(pivot_point=self.pivot_point, sigma=sigma, \
            T=self.plan_T, dt=self.plan_dt)
        
        #
        self.observation_space = Space(
            low=np.array([-10.0, -10.0, -10.0, -2*np.pi, -2*np.pi, -2*np.pi, -10.0, -10.0, -10.0]),
            high=np.array([10.0, 10.0, 10.0, 2*np.pi, 2*np.pi, 2*np.pi, 10.0, 10.0, 10.0]),
        )

        self.action_space = Space(
            low=np.array([0.0]),
            high=np.array([2*self.plan_T])
        )

        # reset the environment
        self.t = 0
        self.reset()
    
    def seed(self, seed):
        np.random.seed(seed=seed)
    
    def reset(self, init_theta=None):
        self.t = 0
        # state for ODE
        self.quad_state = self.quad.reset()
        if init_theta is not None:
            self.pend_state = self.pend.reset(init_theta)
        else:
            self.pend_state = self.pend.reset()
        
        # observation, can be part of the state, e.g., postion
        # or a cartesian representation of the state
        quad_obs = self.quad.get_cartesian_state()
        pend_obs = self.pend.get_cartesian_state()
        #
        # obs = quad_obs.tolist() + pend_obs.tolist()
        obs = (quad_obs - pend_obs).tolist()
        
        return obs

    def step(self, u):
        self.t += self.sim_dt
        opt_t = u
        
        #
        plan_pend_traj, pred_pend_traj_cart = self.planner.plan(self.pend_state, opt_t)
        pred_pend_traj_cart = np.array(pred_pend_traj_cart)
        
        #
        quad_s0 = self.quad_state.tolist()
        ref_traj = quad_s0 + plan_pend_traj + self.quad_sT

        # run nonliear model predictive control
        quad_act, pred_traj = self.nmpc.solve(ref_traj)

        # run the actual control command on the quadrotor
        self.quad_state = self.quad.run(quad_act)
        # simulate one step pendulum
        self.pend_state = self.pend.run()
        
        # update the observation.
        quad_obs = self.quad.get_cartesian_state()
        pend_obs = self.pend.get_cartesian_state()
        obs = (quad_obs - pend_obs).tolist()
        #
        info = {
            "quad_obs": quad_obs, 
            "quad_act": quad_act, 
            "quad_axes": self.quad.get_axes(),
            "pend_obs": pend_obs,
            "pend_corners": self.pend.get_3d_corners(),
            "pred_quad_traj": pred_traj, 
            "pred_pend_traj": pred_pend_traj_cart, 
            "opt_t": opt_t, "plan_dt": self.plan_dt}
        done = False
        if self.t >= (self.sim_T-self.sim_dt):
            done = True

        return obs, 0, done, info
    
    def episode(self, u):
        opt_t = u
        #
        plan_pend_traj, pred_pend_traj_cart = self.planner.plan(self.pend_state, opt_t)
        pred_pend_traj_cart = np.array(pred_pend_traj_cart)
        
        #
        quad_s0 = self.quad_state.tolist()
        ref_traj = quad_s0 + plan_pend_traj + self.quad_sT
    
        _, pred_traj = self.nmpc.solve(ref_traj)
        
        opt_node = np.clip( int(opt_t/self.plan_dt), 0, pred_traj.shape[0]-1)
        # if quad_s0[kPosX] >= self.pivot_point[kPosX]+0.5:
        #     # obs = self.reset()
        #     loss = np.linalg.norm(pred_traj[opt_node, kPosX:kPosZ+1] - np.tile(self.goal_point, (pred_traj.shape[0], 1)))
        #     rew = - np.mean(loss)             
        # else:    
        # opt_min = np.clip(opt_node-10, 0, pred_traj.shape[0]-1)
        # opt_max = np.clip(opt_node+5, 0, pred_traj.shape[0]-1)
        opt_min = np.clip(opt_node-1, 0, pred_traj.shape[0]-1)
        opt_max = np.clip(opt_node+1, 0, pred_traj.shape[0]-1)
        #
        loss = np.linalg.norm(pred_traj[opt_min:opt_max, kPosX:kPosZ+1]  - pred_pend_traj_cart[opt_min:opt_max, kPosX:kPosZ+1])
        rew = - loss
        #    
        return rew, opt_node
    
    @staticmethod
    def _is_within_gap(gap_corners, point):
        A, B, C = [], [], []    
        for i in range(len(gap_corners)):
            p1 = gap_corners[i]
            p2 = gap_corners[(i + 1) % len(gap_corners)]
            
            # calculate A, B and C
            a = -(p2.y - p1.y)
            b = p2.x - p1.x
            c = -(a * p1.x + b * p1.y)

            A.append(a)
            B.append(b)
            C.append(c)
        D = []
        for i in range(len(A)):
            d = A[i] * point.x + B[i] * point.y + C[i]
            D.append(d)

        t1 = all(d >= 0 for d in D)
        t2 = all(d <= 0 for d in D)
        return t1 or t2

    def close(self,):
        return True

    def render(self,):
        return False
    