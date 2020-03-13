import os
import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#
from functools import partial

#
from high_mpc.simulation.dynamic_gap import DynamicGap
from high_mpc.mpc.mpc import MPC
from high_mpc.simulation.animation import SimVisual
import high_mpc.common.logger as logger
#

def run_animate(env):
    #
    env.reset()
    t, n = 0, 0
    while t < env.sim_T:
        t = env.sim_dt * n
        _, _, _, info = env.step()
        #
        n += 1
        update = False
        if t>= env.sim_T:
            update = True
        yield [info, t, update]

def main():
    #
    plan_T = 2.0   # Prediction horizon for MPC and local planner
    plan_dt = 0.04 # Sampling time step for MPC and local planner
    so_path = "./saved/mpc_model/mpc.so"
    #
    mpc = MPC(T=plan_T, dt=plan_dt, so_path=so_path)
    env = DynamicGap(mpc, plan_T, plan_dt)
    
    #
    sim_visual = SimVisual(env)
    run_frame = partial(run_animate, env)
    ani = animation.FuncAnimation(sim_visual.fig, sim_visual.update, frames=run_frame,
            init_func=sim_visual.init_animate, interval=100, blit=True, repeat=False)
    # #
    # writer = animation.writers["ffmpeg"]
    # writer = writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save("MPC_0.mp4", writer=writer)
    # #
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()