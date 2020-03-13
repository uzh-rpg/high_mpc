"""
An animation file for the visulization of the environment
"""
import numpy as np
#
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle, PathPatch, Rectangle
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib.backends.backend_agg import FigureCanvasAgg
#
from high_mpc.common.quad_index import *

#
class SimVisual(object):
    """
    An animation class
    """
    def __init__(self, env):
        #
        self.act_min, self.act_max = -7, 21
        self.pos_min, self.pos_max = -4.0, 7.0
        self.vel_min, self.vel_max = -6.0, 6.0
        self.att_min, self.att_max = -np.pi, np.pi
        self.t_min, self.t_max = 0, env.sim_T
        self.goal_point = env.goal_point
        self.pivot_point = env.pivot_point          # e.g., np.array([2.0, 0.0, 2.0])
        self.frames = []
        #
        # create figure
        self.fig = plt.figure(figsize=(20, 10))
        # and figure grid
        self.gs = gridspec.GridSpec(nrows=4, ncols=10)
        
        # Create layout of our plots
        self.ax_pos = self.fig.add_subplot(self.gs[0, :3])
        self.ax_pos.set_ylim([self.pos_min, self.pos_max])
        self.ax_pos.set_xlim([self.t_min, self.t_max])
        self.ax_pos.legend()

        self.ax_vel = self.fig.add_subplot(self.gs[1, :3])
        self.ax_vel.set_ylim([self.vel_min, self.vel_max])
        self.ax_vel.set_xlim([self.t_min, self.t_max])
        self.ax_vel.legend()

        self.ax_att = self.fig.add_subplot(self.gs[2, :3])
        self.ax_att.set_ylim([self.att_min, self.att_max])
        self.ax_att.set_xlim([0, self.t_max])
        self.ax_att.legend()
        
        self.ax_act = self.fig.add_subplot(self.gs[3, :3])
        self.ax_act.set_ylim([self.act_min, self.act_max])
        self.ax_act.set_xlim([0, self.t_max])
        self.ax_act.legend()

        #
        self.ax_3d = self.fig.add_subplot(self.gs[1:, 3:], projection='3d')
        self.ax_3d.set_xlim([-4, 5])
        self.ax_3d.set_ylim([-2, 2])
        self.ax_3d.set_zlim([ 0, 4])
        self.ax_3d.set_xlabel("x")
        self.ax_3d.set_ylabel("y")
        self.ax_3d.set_zlabel("z")
        
        # Creat "matplotlib lines" for positions.....
        self.l_quad_px, = self.ax_pos.plot([], [], '-r', label='quad_x')
        self.l_pend_px, = self.ax_pos.plot([], [], '--r', label='pend_x')
        self.l_quad_py, = self.ax_pos.plot([], [], '-g', label='quad_y')
        self.l_pend_py, = self.ax_pos.plot([], [], '--g', label='pend_y')
        self.l_quad_pz, = self.ax_pos.plot([], [], '-b', label='quad_z')
        self.l_pend_pz, = self.ax_pos.plot([], [], '--b', label='pend_z')
        

        # Creat "matplotlib lines" for velocity.....
        self.l_quad_vx, = self.ax_vel.plot([], [], '-r', label='quad_vx')
        self.l_pend_vx, = self.ax_vel.plot([], [], '--r', label='pend_vx')
        self.l_quad_vy, = self.ax_vel.plot([], [], '-g', label='quad_vy')
        self.l_pend_vy, = self.ax_vel.plot([], [], '--g', label='pend_vy')
        self.l_quad_vz, = self.ax_vel.plot([], [], '-b', label='quad_vz')
        self.l_pend_vz, = self.ax_vel.plot([], [], '--b', label='pend_vz')
        
        # Creat "matplotlib lines" for attitude.....
        self.l_quad_ax, = self.ax_att.plot([], [], '-r', label='quad_ax')
        self.l_pend_ax, = self.ax_att.plot([], [], '--r', label='pend_ax')
        self.l_quad_ay, = self.ax_att.plot([], [], '-g', label='quad_ay')
        self.l_pend_ay, = self.ax_att.plot([], [], '--g', label='pend_ay')
        self.l_quad_az, = self.ax_att.plot([], [], '-b', label='quad_az')
        self.l_pend_az, = self.ax_att.plot([], [], '--b', label='pend_az')

        # Creat "matplotlib lines" for actions.....
        self.l_quad_thrust, = self.ax_act.plot([], [], '-k', label='quad_thrust')
        self.l_quad_wx, = self.ax_act.plot([], [], '-r', label='quad_wx')
        self.l_quad_wy, = self.ax_act.plot([], [], '-g', label='quad_wy')
        self.l_quad_wz, = self.ax_act.plot([], [], '-b', label='quad_wz')
        
        # Plot 3D coordinates
        self.l_quad_pos, = self.ax_3d.plot([], [], [], 'b-')
        self.l_quad_pred_traj, = self.ax_3d.plot([], [], [], 'r*')
        self.l_pend_pred_traj, = self.ax_3d.plot([], [], [], 'ko')
        #
        self.l_pend, = self.ax_3d.plot([], [], [], 'b', linewidth=2)
        self.l_pend_edge1, = self.ax_3d.plot([], [], [], 'b', linewidth=2)
        self.l_pend_edge2, = self.ax_3d.plot([], [], [], 'b', linewidth=2)
        self.l_pend_edge3, = self.ax_3d.plot([], [], [], 'b', linewidth=2)
        self.l_pend_edge4, = self.ax_3d.plot([], [], [], 'b', linewidth=2)
        #
        self.l_quad_x, = self.ax_3d.plot([], [], [], 'r', linewidth=3)
        self.l_quad_y, = self.ax_3d.plot([], [], [], 'g', linewidth=3)
        self.l_quad_z, = self.ax_3d.plot([], [], [], 'b', linewidth=3)
        
        #
        self.ax_3d.scatter(self.pivot_point[0], self.pivot_point[1], self.pivot_point[2], marker='o', color='r')
        self.ax_3d.scatter(self.goal_point[0], self.goal_point[1], self.goal_point[2], marker='o', color='r', s=5)
        self.ax_3d.view_init(elev=20, azim=110)
        # Draw a circle on the x=0 'wall'

        # # Ground
        width, height = 5, 2
        g = Rectangle(xy=(0.5-width, 0-height), width=2*width, height=2*height, \
            alpha=0.8, facecolor='gray', edgecolor='black')
        self.ax_3d.add_patch(g)
        art3d.pathpatch_2d_to_3d(g, z=0, zdir="z")
        #
        # #
        self.reset_buffer()
        
    def reset_buffer(self, ):
        #
        self.ts = []
        self.quad_pos, self.quad_vel, self.quad_att, self.quad_cmd = [], [], [], []
        self.pend_pos, self.pend_vel, self.pend_att = [], [], []
        self.quad_hist = []

    def init_animate(self,):
        # Initialize position.....
        self.l_quad_px.set_data([], [])
        self.l_pend_px.set_data([], [])
        self.l_quad_py.set_data([], [])
        self.l_pend_py.set_data([], [])
        self.l_quad_pz.set_data([], [])
        self.l_pend_pz.set_data([], [])

        # Initialize velocity.....
        self.l_quad_vx.set_data([], [])
        self.l_pend_vx.set_data([], [])
        self.l_quad_vy.set_data([], [])
        self.l_pend_vy.set_data([], [])
        self.l_quad_vz.set_data([], [])
        self.l_pend_vz.set_data([], [])

        # Initialize attitude.....
        self.l_quad_ax.set_data([], [])
        self.l_pend_ax.set_data([], [])
        self.l_quad_ay.set_data([], [])
        self.l_pend_ay.set_data([], [])
        self.l_quad_az.set_data([], [])
        self.l_pend_az.set_data([], [])

        # Initialize attitude.....
        self.l_quad_thrust.set_data([], [])
        self.l_quad_wx.set_data([], [])
        self.l_quad_wy.set_data([], [])
        self.l_quad_wz.set_data([], [])
        
        # Initialize quadrotor 3d trajectory
        self.l_quad_pos.set_data([], [])
        self.l_quad_pos.set_3d_properties([])
        # Initialize MPC planned trajectory
        self.l_quad_pred_traj.set_data([], [])
        self.l_quad_pred_traj.set_3d_properties([])
        # Initialize planned pendulum trajectory
        self.l_pend_pred_traj.set_data([], [])
        self.l_pend_pred_traj.set_3d_properties([])
        # Initialize pendulum plot
        self.l_pend.set_data([], [])
        self.l_pend.set_3d_properties([])
        #
        self.l_pend_edge1.set_data([], [])
        self.l_pend_edge1.set_3d_properties([])
        self.l_pend_edge2.set_data([], [])
        self.l_pend_edge2.set_3d_properties([])
        self.l_pend_edge3.set_data([], [])
        self.l_pend_edge3.set_3d_properties([])
        self.l_pend_edge4.set_data([], [])
        self.l_pend_edge4.set_3d_properties([])

        # Initialize quad arm
        self.l_quad_x.set_data([], [])
        self.l_quad_x.set_3d_properties([])
        self.l_quad_y.set_data([], [])
        self.l_quad_y.set_3d_properties([])
        self.l_quad_z.set_data([], [])
        self.l_quad_z.set_3d_properties([])
        #
        return self.l_quad_px, self.l_quad_py, self.l_quad_pz, \
            self.l_quad_vx, self.l_quad_vy, self.l_quad_vz, \
            self.l_quad_ax, self.l_quad_ay, self.l_quad_az, \
            self.l_quad_thrust, self.l_quad_wx, self.l_quad_wy, self.l_quad_wz, \
            self.l_pend_px, self.l_pend_py, self.l_pend_pz, \
            self.l_pend_vx, self.l_pend_vy, self.l_pend_vz, \
            self.l_pend_ax, self.l_pend_ay, self.l_pend_az, \
            self.l_quad_pos, self.l_quad_pred_traj, self.l_pend, \
            self.l_pend_pred_traj, self.l_quad_x, self.l_quad_y,  self.l_quad_z, \
            self.l_pend_edge1, self.l_pend_edge2, self.l_pend_edge3, self.l_pend_edge4, 

    def update(self, data_info):
        info, t, update = data_info[0], data_info[1], data_info[2]
        quad_obs = info["quad_obs"]
        quad_act = info["quad_act"]
        quad_axes = info["quad_axes"] 
        pend_obs = info["pend_obs"]
        pend_corners = info["pend_corners"]
        pred_quad_traj = info["pred_quad_traj"]
        pred_pend_traj = info["pred_pend_traj"]
        opt_t = info["opt_t"]
        plan_dt = info["plan_dt"]
        opt_idx = np.clip( int(opt_t/plan_dt), 0, pred_quad_traj.shape[0]-1)
        
        if update:
            self.reset_buffer()
        else:
            self.ts.append(t)
            #
            self.quad_pos.append(quad_obs[0:3])
            self.quad_att.append(quad_obs[3:6])
            self.quad_vel.append(quad_obs[6:9])
            self.quad_cmd.append(quad_act[0:4])
            #
            self.pend_pos.append(pend_obs[0:3])
            self.pend_att.append(pend_obs[3:6])
            self.pend_vel.append(pend_obs[6:9])

        if len(self.ts) == 0:
            self.init_animate()
        else:
            quad_pos_arr = np.array(self.quad_pos)
            self.l_quad_px.set_data(self.ts, quad_pos_arr[:, 0])
            self.l_quad_py.set_data(self.ts, quad_pos_arr[:, 1])
            self.l_quad_pz.set_data(self.ts, quad_pos_arr[:, 2])
            #
            quad_vel_arr = np.array(self.quad_vel)
            self.l_quad_vx.set_data(self.ts, quad_vel_arr[:, 0])
            self.l_quad_vy.set_data(self.ts, quad_vel_arr[:, 1])
            self.l_quad_vz.set_data(self.ts, quad_vel_arr[:, 2])
            #
            quad_att_arr = np.array(self.quad_att)
            self.l_quad_ax.set_data(self.ts, quad_att_arr[:, 0])
            self.l_quad_ay.set_data(self.ts, quad_att_arr[:, 1])
            self.l_quad_az.set_data(self.ts, quad_att_arr[:, 2])
            # 
            quad_act_arr = np.array(self.quad_cmd)
            self.l_quad_thrust.set_data(self.ts, quad_act_arr[:, 0])
            self.l_quad_wx.set_data(self.ts, quad_act_arr[:, 1])
            self.l_quad_wy.set_data(self.ts, quad_act_arr[:, 2])
            self.l_quad_wz.set_data(self.ts, quad_act_arr[:, 3])
                    
            #
            pend_pos_arr = np.array(self.pend_pos)
            self.l_pend_px.set_data(self.ts, pend_pos_arr[:, 0])
            self.l_pend_py.set_data(self.ts, pend_pos_arr[:, 1])
            self.l_pend_pz.set_data(self.ts, pend_pos_arr[:, 2])
            #
            pend_vel_arr = np.array(self.pend_vel)
            self.l_pend_vx.set_data(self.ts, pend_vel_arr[:, 0])
            self.l_pend_vy.set_data(self.ts, pend_vel_arr[:, 1])
            self.l_pend_vz.set_data(self.ts, pend_vel_arr[:, 2])
            #
            pend_att_arr = np.array(self.pend_att)
            self.l_pend_ax.set_data(self.ts, pend_att_arr[:, 0])
            self.l_pend_ay.set_data(self.ts, pend_att_arr[:, 1])
            self.l_pend_az.set_data(self.ts, pend_att_arr[:, 2])
            

            # plot quadrotor trajectory
            self.l_quad_pos.set_data(quad_pos_arr[:, 0], quad_pos_arr[:, 1])
            self.l_quad_pos.set_3d_properties(quad_pos_arr[:, 2])
            # plot mpc plan trajectory
            self.l_quad_pred_traj.set_data(pred_quad_traj[:, 0], pred_quad_traj[:, 1])
            self.l_quad_pred_traj.set_3d_properties(pred_quad_traj[:, 2])
            if quad_pos_arr[-1, 0] <= 2.0:
                # plot planner trajectory
                self.l_pend_pred_traj.set_data(pred_pend_traj[opt_idx, 0], pred_pend_traj[opt_idx, 1])
                self.l_pend_pred_traj.set_3d_properties(pred_pend_traj[opt_idx, 2])
            
            #
            self.l_pend.set_data([self.pivot_point[0], pend_obs[kPosX]], [self.pivot_point[1], pend_obs[kPosY]] )
            self.l_pend.set_3d_properties([self.pivot_point[2], pend_obs[kPosZ]])
                     
            c1, c2, c3, c4 = pend_corners
            #
            self.l_pend_edge1.set_data([c1[0], c2[0]], [c1[1], c2[1]])
            self.l_pend_edge1.set_3d_properties([c1[2], c2[2]])
            self.l_pend_edge2.set_data([c2[0], c2[0]], [c2[1], c4[1]])
            self.l_pend_edge2.set_3d_properties([c2[2], c4[2]])
            self.l_pend_edge3.set_data([c4[0], c3[0]], [c4[1], c3[1]])
            self.l_pend_edge3.set_3d_properties([c4[2], c3[2]])
            self.l_pend_edge4.set_data([c3[0], c1[0]], [c3[1], c1[1]])
            self.l_pend_edge4.set_3d_properties([c3[2], c1[2]])
            
            self.quad_hist.append(quad_axes)
            #
            for _, quad_motor in enumerate(self.quad_hist):
                axes_x, axes_y, axes_z  = quad_motor
                self.l_quad_x.set_data([quad_pos_arr[-1, 0], axes_x[0]], [quad_pos_arr[-1, 1], axes_x[1]])
                self.l_quad_x.set_3d_properties([quad_pos_arr[-1, 2], axes_x[2]])
                self.l_quad_y.set_data([quad_pos_arr[-1, 0], axes_y[0]], [quad_pos_arr[-1, 1], axes_y[1]])
                self.l_quad_y.set_3d_properties([quad_pos_arr[-1, 2], axes_y[2]])
                self.l_quad_z.set_data([quad_pos_arr[-1, 0], axes_z[0]], [quad_pos_arr[-1, 1], axes_z[1]])
                self.l_quad_z.set_3d_properties([quad_pos_arr[-1, 2], axes_z[2]])

        return self.l_quad_px, self.l_quad_py, self.l_quad_pz, \
            self.l_quad_vx, self.l_quad_vy, self.l_quad_vz, \
            self.l_quad_ax, self.l_quad_ay, self.l_quad_az, \
            self.l_quad_thrust, self.l_quad_wx, self.l_quad_wy, self.l_quad_wz, \
            self.l_pend_px, self.l_pend_py, self.l_pend_pz, \
            self.l_pend_vx, self.l_pend_vy, self.l_pend_vz, \
            self.l_pend_ax, self.l_pend_ay, self.l_pend_az, \
            self.l_quad_pos, self.l_quad_pred_traj, self.l_pend, \
            self.l_pend_pred_traj, self.l_quad_x, self.l_quad_y,  self.l_quad_z, \
            self.l_pend_edge1, self.l_pend_edge2, self.l_pend_edge3, self.l_pend_edge4, 
    