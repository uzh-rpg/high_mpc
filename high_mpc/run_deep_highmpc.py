import os
import datetime
import argparse
import numpy as np
from functools import partial

# A Gym style environment
from autompc.env.dynamic_gap_v0 import DynamicGap_v0
#
from autompc.common.animate import NMPCDemo

# reinforcement learning ..
import high_mpc.imitation_learning.imitation as Imitation
import high_mpc.rl.common.util as U
import high_mpc.rl.common.logger as logger
#

def parser_il():
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', type=int, default=0,
        help="To train new model or simply test pre-trained model")
    parser.add_argument('--save_dir', type=str, default=os.path.dirname(os.path.realpath(__file__)),
        help="Directory where to save the checkpoints and training metrics")
    parser.add_argument('--env', type=str, default="DynamicGap_v0", help="Environment Name")
    parser.add_argument('--load_dir', type=str, help="Directory where to load weights")
    return parser
    
def test_sac(env, actor_params, load_dir):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = env.action_space.high
    min_action = env.action_space.low
    max_ep_length = env.max_episode_steps
    #
    actor = Imitation.Actor(obs_dim, act_dim)
    actor.load_weights(load_dir)
    
    #
    avg_rewards = 0.0
    ep_len, ep_reward =  0, 0
    for i in range(10):
        obs = env.reset()
        t = 0
        while t < env.sim_T:
            t += env.sim_dt
            # if obs[0] >= 0.1: # 
            #     act = np.array( [10.0])
            # else:
            # sample action either randomly or from the stochastic actor
            # based on current time step
            obs_tmp = np.reshape(obs, (1, -1)) # to please tensorflow
            act = actor(obs_tmp).numpy()[0]

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
    args = parser_il().parse_args()
    env = DynamicGap_v0(so_path="/home/sysadmin/GitHub/rpg_auto_mpc/main/nmpc_v0.so")

    #
    actor_params = dict(
        hidden_units=[32, 32],
        learning_rate=1e-4,
        activation='relu',
        train_epoch=1000,
        batch_size=128
    )

    #
    il_params = dict(
        max_samples =5000,
        max_wml_iter=15,
        beta0=3.0,
        n_samples=15,
    )
    
    # if in training mode, create new dir to save the model and checkpoints
    if args.option == 0: # collect data
        save_dir = U.get_dir(args.save_dir + "/save_model")
        save_dir = U.get_dir(save_dir + "/" + args.env + "/")
        save_dir = os.path.join(save_dir, datetime.datetime.now().strftime("il-%m-%d-%H-%M-%S"))

        #
        logger.configure(dir=save_dir)
        logger.log("***********************Log & Store Hyper-parameters***********************")
        logger.log("actor params")
        logger.log(actor_params)
        logger.log("sac params")
        logger.log(il_params)
        logger.log("***************************************************************************")

        # run soft actor-critic algorithm
        Imitation.data_collection(env=env, logger=logger, \
            save_dir=save_dir, **il_params)

    elif args.option == 1: # train the policy
        data_dir = "/home/sysadmin/GitHub/rpg_auto_mpc/main/save_model/Dataset2"
        Imitation.train(env, logger, data_dir, **actor_params)

    elif args.option == 2: # evaluate the policy
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        load_dir = "/home/sysadmin/GitHub/rpg_auto_mpc/main/save_model/Dataset2/act_net/weights_999.h5"
        demo = NMPCDemo(env.goal_point, env.pivot_point, t_max=env.sim_T)
        run_frame = partial(test_sac, env, actor_params, load_dir)
        ani = animation.FuncAnimation(demo.fig, demo.update, frames=run_frame, 
            init_func=demo.init_animate, interval=100, blit=True, repeat=False)
        # #
        writer = animation.writers["ffmpeg"]
        writer = writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
        ani.save("output.mp4", writer=writer)
        # #
        plt.tight_layout()
        # plt.axis('off')
        plt.show()
        
if __name__ == "__main__":
    main()