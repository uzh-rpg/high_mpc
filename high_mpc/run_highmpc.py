import os
import datetime
import argparse
import numpy as np
from functools import partial

# A Gym style environment
from high_mpc.simulation.dynamic_gap import DynamicGap_v0
##
from autompc.common.animate import NMPCDemo

# reinforcement learning ..
import autompc.rl.algos.wml as Wml
import autompc.rl.common.util as U
import autompc.rl.common.logger as logger

def parser_sac():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=1,
        help="To train new model or simply test pre-trained model")
    parser.add_argument('--save_dir', type=str, default=os.path.dirname(os.path.realpath(__file__)),
        help="Directory where to save the checkpoints and training metrics")
    parser.add_argument('--seed', type=int, default=2349, help="Random seed")
    parser.add_argument('--beta', type=float, default=3.0, help="beta")
    parser.add_argument('--env', type=str, default="DynamicGap_v0", help="Environment Name")
    parser.add_argument('--load_dir', type=str, help="Directory where to load weights")
    return parser

def test_wml(env, load_dir):
    act_dim = env.action_space.shape[0]
    clip_value = [0, env.plan_T]
    #
    pi = Wml.GaussianPolicy_v0(act_dim, clip_value=clip_value)
    
    #
    pi.load_dir(load_dir)
    ep_len, ep_reward =  0, 0

    for i in range(10):
        _ = env.reset()
        t = 0
        while t < env.sim_T:
            t += env.sim_dt
            # sample action either randomly or from the stochastic actor
            # based on current time step
            act = pi()
        
            # execute action
            next_obs, reward, done, info = env.step(act)

            #
            obs = next_obs
            ep_reward += reward

            #
            ep_len += 1

            #
            update = False
            if t >= env.sim_T:
                update = True
            yield [info, t, update]

def main():
    #
    args = parser_sac().parse_args()
    env = DynamicGap_v0(so_path="/home/sysadmin/GitHub/rpg_auto_mpc/main/nmpc_v0.so")
    U.set_global_seed(args.seed)
    #
    wml_params = dict(
        sigma0=100,
        max_iter=20,
        n_samples=20,
        beta0=args.beta,
    )

    if args.train:
        save_dir = U.get_dir(args.save_dir + "/wml")
        save_dir = U.get_dir(save_dir + "/" + args.env)
        save_dir = U.get_dir(save_dir + "/" + str(args.beta))
        save_dir = U.get_dir(save_dir + "/" + str(args.seed) + "/")
        save_dir = os.path.join(save_dir, datetime.datetime.now().strftime("wml-%m-%d-%H-%M-%S"))

                #
        logger.configure(dir=save_dir)
        logger.log("***********************Log & Store Hyper-parameters***********************")
        logger.log("locally wml params")
        logger.log(wml_params)
        logger.log("***************************************************************************")
        Wml.run_wml(env=env, logger=logger, save_dir=save_dir, **wml_params)
    else:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        load_dir = "/home/sysadmin/GitHub/rpg_auto_mpc/main/wml/DynamicGap_v0/3.0/2349/wml-02-29-19-14-36/actor_net/weights_19.h5.npz"
        demo = NMPCDemo(env.goal_point, env.pivot_point, t_max=env.sim_T)
        run_frame = partial(test_wml, env,  load_dir)
        ani = animation.FuncAnimation(demo.fig, demo.update, frames=run_frame, 
            init_func=demo.init_animate, interval=100, blit=True, repeat=False)
        
        #
        plt.tight_layout()
        plt.show()
if __name__ == "__main__":
    main()