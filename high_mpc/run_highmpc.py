import os
import datetime
import argparse
import numpy as np
from functools import partial
#
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# A Gym style environment
from high_mpc.simulation.dynamic_gap import DynamicGap
from high_mpc.mpc.high_mpc import High_MPC
from high_mpc.simulation.animation import SimVisual
from high_mpc.common import logger
from high_mpc.common import util as U
from high_mpc.policy import high_policy
#

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default=os.path.dirname(os.path.realpath(__file__)),
        help="Directory where to save the checkpoints and training metrics")
    parser.add_argument('--seed', type=int, default=2349, help="Random seed")
    parser.add_argument('--beta', type=float, default=3.0, help="beta")
    return parser

def main():
    #
    args = arg_parser().parse_args()
    #
    plan_T = 2.0   # Prediction horizon for MPC and local planner
    plan_dt = 0.04 # Sampling time step for MPC and local planner
    so_path = "./mpc/saved/high_mpc.so" # saved high mpc model (casadi code generation)
    
    #
    high_mpc = High_MPC(T=plan_T, dt=plan_dt, so_path=so_path)
    env = DynamicGap(high_mpc, plan_T, plan_dt)
    
    #
    U.set_global_seed(args.seed)
    
    #
    wml_params = dict(
        sigma0=100,
        max_iter=20,
        n_samples=20,
        beta0=args.beta,
    )

    save_dir = U.get_dir(args.save_dir + "/saved_policy")
    save_dir = os.path.join(save_dir, datetime.datetime.now().strftime("highmpc-%m-%d-%H-%M-%S"))

    #
    logger.configure(dir=save_dir)
    logger.log("***********************Log & Store Hyper-parameters***********************")
    logger.log("weighted maximum likelihood params")
    logger.log(wml_params)
    logger.log("***************************************************************************")
    high_policy.run_wml(env=env, logger=logger, save_dir=save_dir, **wml_params)

if __name__ == "__main__":
    main()