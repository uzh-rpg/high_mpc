import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.animation as animation
import matplotlib.pyplot as plt
#
def get_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

#
def set_global_seed(seed, env=None):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if env is not None:
        env.seed(seed)

def tf_config_gpu_memory():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def tf_config_gpu_memory_02():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

#
def merge_dicts(*dicts):
    d = {}
    for dic in dicts:
        for key in dic:
            try:
                d[key].append(dic[key])
            except KeyError:
                d[key] = [dic[key]]
    for key in d:
        d[key] = np.concatenate(d[key]).squeeze()
    return d
#
def test_run(env, actor, num_rollouts, render=True):
    max_ep_length = env.max_episode_steps
    # video = get_dir("./video")
    # env = gym_wrappers.Monitor(env, video, force=True)
    avg_rewards = 0.0
    for _ in range(num_rollouts):
        obs, done, ep_length = env.reset(), False, 0
        while not (done or (ep_length == max_ep_length)):
            if render:
                env.render()
            obs_tmp = np.reshape(obs, (1, -1))
            act = actor.step(obs_tmp, stochastic=False).numpy()[0]
            #
            obs, rew, done, _ = env.step(act)
            #
            avg_rewards += rew
            ep_length += 1
    env.close()
    return avg_rewards / num_rollouts

# def plot_reward(logger_dir, title, var_name=["ep_reward", "entropy"]):
#     csv_path = logger_dir + "/progress.csv"
#     fig, ax = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
#     data = pd.read_csv(csv_path)
#     #
#     sns.tsplot(data=data[var_name[0]].dropna(), ax=ax[0])
#     sns.tsplot(data=data[var_name[1]].dropna(), ax=ax[1])
#     ax[0].set_ylabel(var_name[0])
#     ax[1].set_ylabel(var_name[1])
#     fig.suptitle(title)
#     plt.show()


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y