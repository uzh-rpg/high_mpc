import os
import datetime
import argparse
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# A Gym style environment
from high_mpc.simulation.dynamic_gap import DynamicGap
from high_mpc.mpc.high_mpc import High_MPC
from high_mpc.simulation.animation import SimVisual
from high_mpc.common import logger
from high_mpc.common import util as U
from high_mpc.policy import deep_high_policy

#
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', type=int, default=0,
        help="0 - Data collection; 1 - train the deep high-level policy; 2 - test the trained policy.")
    parser.add_argument('--save_dir', type=str, default=os.path.dirname(os.path.realpath(__file__)),
        help="Directory where to save the checkpoints and training metrics")
    parser.add_argument('--save_video', type=bool, default=False,
        help="Save the animation as a video file")
    parser.add_argument('--load_dir', type=str, help="Directory where to load weights")
    return parser
    
def run_deep_high_mpc(env, actor_params, load_dir):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    #
    actor = deep_high_policy.Actor(obs_dim, act_dim)
    actor.load_weights(load_dir)
    #
    ep_len, ep_reward =  0, 0
    for i in range(10):
        obs = env.reset()
        t = 0
        while t < env.sim_T:
            t += env.sim_dt
            #
            obs_tmp = np.reshape(obs, (1, -1)) # to please tensorflow
            act = actor(obs_tmp).numpy()[0]

            # execute action
            next_obs, reward, _, info = env.step(act)

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
    args = arg_parser().parse_args()
    #
    plan_T = 2.0   # Prediction horizon for MPC and local planner
    plan_dt = 0.04 # Sampling time step for MPC and local planner
    so_path = "./mpc/saved/high_mpc.so" # saved high mpc model (casadi code generation)
    #
    high_mpc = High_MPC(T=plan_T, dt=plan_dt, so_path=so_path)
    env = DynamicGap(high_mpc, plan_T, plan_dt)

    #
    actor_params = dict(
        hidden_units=[32, 32],
        learning_rate=1e-4,
        activation='relu',
        train_epoch=1000,
        batch_size=128
    )

    # training
    training_params = dict(
        max_samples =5000,
        max_wml_iter=15,
        beta0=3.0,
        n_samples=15,
    )
    
    # if in training mode, create new dir to save the model and checkpoints
    if args.option == 0: # collect data
        save_dir = U.get_dir(args.save_dir + "/Dataset")
        save_dir = os.path.join(save_dir, datetime.datetime.now().strftime("deep_highmpc-%m-%d-%H-%M-%S"))

        #
        logger.configure(dir=save_dir)
        logger.log("***********************Log & Store Hyper-parameters***********************")
        logger.log("actor params")
        logger.log(actor_params)
        logger.log("training params")
        logger.log(training_params)
        logger.log("***************************************************************************")

        # 
        deep_high_policy.data_collection(env=env, logger=logger, \
            save_dir=save_dir, **training_params)
    elif args.option == 1: # train the policy
        data_dir = args.save_dir + "/Dataset"
        deep_high_policy.train(env, logger, data_dir, **actor_params)
    elif args.option == 2: # evaluate the policy
        load_dir = args.save_dir + "/Dataset/act_net/weights_999.h5"
        sim_visual = SimVisual(env)
        run_frame = partial(run_deep_high_mpc, env, actor_params, load_dir)
        ani = animation.FuncAnimation(sim_visual.fig, sim_visual.update, frames=run_frame, 
            init_func=sim_visual.init_animate, interval=100, blit=True, repeat=False)
        # #
        if args.save_video:
            writer = animation.writers["ffmpeg"]
            writer = writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
            ani.save("output.mp4", writer=writer)
        # #
        plt.tight_layout()
        # plt.axis('off')
        plt.show()
        
if __name__ == "__main__":
    main()