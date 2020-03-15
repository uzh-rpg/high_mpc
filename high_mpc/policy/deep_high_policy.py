"""
Multi-layer perception policy
"""

import os
import numpy as np
import glob
#
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
#
from high_mpc.policy.wml import GaussianPolicy_v0

class Actor(Model):

    def __init__(self, obs_dim, act_dim, \
        learning_rate=3e-4, hidden_units=[32, 32],  \
        activation='relu', trainable=True, actor_name="actor"):
        super(Actor, self).__init__(actor_name)
        #
        obs = Input(shape=(obs_dim, ))
        #
        x = Dense(hidden_units[0], activation=activation, trainable=trainable)(obs)
        x = Dense(hidden_units[1], activation=activation, trainable=trainable)(x)
        #
        act = Dense(act_dim, trainable=trainable)(x)
        #
        self._act_net = Model(inputs=obs, outputs=act)
        self._optimizer = Adam(lr=learning_rate)

    def call(self, inputs):
        return self._act_net(inputs)

    def save_weights(self, save_dir, iter):
        save_dir = save_dir + "/act_net"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        weights_path = save_dir + "/weights_{0}.h5".format(iter)
        self._act_net.save_weights(weights_path)

    def load_weights(self, file_path):
        self._act_net.load_weights(file_path)

    @tf.function
    def train_batch(self, obs_batch, act_batch):
        with tf.GradientTape(persistent=True) as tape:
            act_pred = self.call(obs_batch)

            # mean squared error
            mse_loss = 0.5 * tf.math.reduce_mean( (act_batch - act_pred)**2 )
        #   
        act_grad = tape.gradient(mse_loss, self._act_net.trainable_variables)
        self._optimizer.apply_gradients( zip(act_grad, self._act_net.trainable_variables))

        del tape

        return mse_loss

class Dataset(object):
    
    def __init__(self, obs_dim, act_dim, max_size=int(1e6)):
        #
        self._obs_buf = np.zeros(shape=(max_size, obs_dim), dtype=np.float32)
        self._act_buf = np.zeros(shape=(max_size, act_dim), dtype=np.float32)
        #
        self._max_size = max_size
        self._ptr = 0
        self._size = 0

    def add(self, obs, act):
        self._obs_buf[self._ptr] = obs
        self._act_buf[self._ptr] = act
        #
        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)
    
    def sample(self, batch_size):
        index = np.random.randint(0, self._size, size=batch_size)
        #
        return [ self._obs_buf[index], self._act_buf[index] ]
    
    def save(self, save_dir, n_iter):
        save_dir = save_dir + "/dataset"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        data_path = save_dir + "/data_{0}".format(n_iter)
        np.savez(data_path, obs=self._obs_buf, act=self._act_buf)

    def load(self, data_dir):
        self._obs_buf, self._act_buf = None, None
        for i, data_file in enumerate(glob.glob(os.path.join(data_dir, "*.npz"))):
            size = int(data_file.split("/")[-1].split("_")[-1].split(".")[0])
            np_file = np.load(data_file)
            #
            obs_array = np_file['obs'][:size, :]
            act_array = np_file['act'][:size, :]
            if i == 0:
                self._obs_buf = obs_array
                self._act_buf = act_array    
            else:
                self._obs_buf = np.append(self._obs_buf, obs_array, axis=0)
                self._act_buf = np.append(self._act_buf, act_array, axis=0)
        #
        self._size  = self._obs_buf.shape[0]
        self._ptr = self._size - 1


def data_collection(env, logger, save_dir, max_samples, 
    max_wml_iter, beta0, n_samples):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    #
    act_low = env.action_space.low
    act_high = env.action_space.high
    #
    clip_value = [ act_low, act_high]

    data_set = Dataset(obs_dim, act_dim)
    # Data collection
    obs, done = env.reset(), True
    save_iter = 100
    for n in range(max_samples):
        if done:
            obs = env.reset()
        
        # # # # # # # # # # # # # # # # # # 
        # ---- Weighted Maximum Likelihood 
        # # # # # # # # # # # # # # # # # # 
        pi = GaussianPolicy_v0(act_dim, clip_value=clip_value)
        # online optimization
        opt_success, pass_index = True, 0
        if obs[0] >= 0.2: # 
            act = np.array( [4.0] )
        # elif obs[0] >= -0.4 and obs[0] <= 0.1:
        #     act = np.array( [0.0] )
        else: # weighted maximum likelihood optimization....
            for i in range(max_wml_iter):
                rewards = np.zeros(n_samples)
                Actions = np.zeros(shape=(n_samples, pi.act_dim))
                for j in range(n_samples):
                    act = pi.sample()
                    #
                    rewards[j], pass_index = env.episode(act)
                    #
                    Actions[j, :] = act
                #
                if not (np.max(rewards) - np.min(rewards)) <= 1e-10:
                    # compute weights 
                    beta = beta0 / (np.max(rewards) - np.min(rewards))
                    weights = np.exp( beta * (rewards - np.max(rewards)))
                    Weights = weights / np.mean(weights)
                    # 
                    pi.fit(Weights, Actions)
                    opt_success = True   
                else:
                    opt_success = False     
                #
                logger.log("********** Sample %i ************"%n)
                logger.log("---------- Wml_iter %i ----------"%i)
                logger.log("---------- Reward %f ----------"%np.mean(rewards))
                logger.log("---------- Pass index %i ----------"%pass_index)
                if abs(np.mean(rewards)) <= 0.001:
                    logger.log("---------- Converged %f ----------"%np.mean(rewards))
                    break
        
            # take the optimal value
            if opt_success:
                act = pi()
                logger.log("---------- Success {0} ----------".format(act))
            else:
                act = np.array( [(pass_index+1) * env.plan_dt] )
                logger.log("---------- Faild {0} ----------".format(act))
        # collect optimal action
        data_set.add(obs, act)

        # # # # # # # # # # # # # # # # # # 
        # ---- Execute the action --------
        # # # # # # # # # # # # # # # # # # 
        #
        obs, _, done, _ = env.step(act)
        logger.log("---------- Obs{0}  ----------".format(obs))
        logger.log("                          ")

        #
        if (n+1) % save_iter == 0:
            data_set.save(save_dir, n)

        
def train(env, logger, data_dir, hidden_units, learning_rate, \
     activation, train_epoch, batch_size):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    #
    # dataset = Dataset(obs_dim, act_dim)
    # dataset.load(data_dir)
    train_obs, train_act = None, None
    for i, data_file in enumerate(glob.glob(os.path.join(data_dir, "*.npz"))):
        size = int(data_file.split("/")[-1].split("_")[-1].split(".")[0])
        np_file = np.load(data_file)
        #
        obs_array = np_file['obs'][:size, :]
        act_array = np_file['act'][:size, :]
        if i == 0:
            train_obs = obs_array
            train_act = act_array    
        else:
            train_obs = np.append(train_obs, obs_array, axis=0)
            train_act = np.append(train_act, act_array, axis=0)
    #
    train_tf_dataset = tf.data.Dataset.from_tensor_slices((train_obs, train_act))
    #
    actor = Actor(obs_dim, act_dim)
    #
    SHUFFLE_BUFFER_SIZE = 100
    for epoch in range(train_epoch):
        #
        train_loss = []
        dataset = train_tf_dataset.shuffle(SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True)
        for obs, act in dataset.batch(batch_size, drop_remainder=False):
            loss = actor.train_batch(obs, act)
            train_loss.append(loss)

        if (epoch+1) % 100 == 0:
            print("Epoch {:03d}: Loss: {:.3f}".format(epoch, np.mean(train_loss)))
            actor.save_weights(data_dir, iter=epoch)


    
        




    