"""
Gaussian Policy
"""
import os
import numpy as np

#
class GaussianPolicy_v0(object):
    """
    Gaussian Policy with constant mean, constant variance
    """
    def __init__(self, act_dim, sigma0=100.0, clip_value=None):
        self.act_dim = act_dim
        self.mu = np.array([0.5])
        self.scale = np.diag([1.0]) * sigma0
        #
        self.clip_low = clip_value[0]
        self.clip_high = clip_value[1]
    
    def __call__(self):
        act = self.mu
        return np.clip(act, self.clip_low, self.clip_high)

    def sample(self):
        act = np.random.multivariate_normal(mean=self.mu, cov=self.scale)
        return np.clip(act, self.clip_low, self.clip_high)
        
    def fit(self, Weights, Actions):
        """
        Update policy parameters via Reward-Weighted Maximum Likelyhood
        """
        self.mu = Weights.dot(Actions)/ np.sum(Weights+1e-8)
        Z = (np.sum(Weights)**2 - np.sum(Weights**2)) / np.sum(Weights)
        self.scale =  np.sum([Weights[i]*(np.outer((Actions[i]-self.mu), (Actions[i]-self.mu))) for i in range(len(Weights))], 0)/Z

    def save_weight(self, save_dir, i=0):
        save_dir = save_dir + "/actor_net"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        weights_path = save_dir + "/weights_{0}.h5".format(i)
        np.savez(weights_path, mu=self.mu, scale=self.scale)
    
    def load_dir(self, file_path):
        npzfile = np.load(file_path)
        self.mu = npzfile['mu']
        self.scale = npzfile['scale'] 

def run_wml(env, logger, save_dir, n_samples, max_iter, sigma0, beta0=3.0):
    """
    Weighted Maximum Likelihood (Gaussian Policy, Constant mean)
    """
    #
    act_dim = env.action_space.shape[0]
    #
    act_low = env.action_space.low
    act_high = env.action_space.high
    #
    clip_value = [ act_low, act_high]
    
    # -- Training
    obs = env.reset()
    pi = GaussianPolicy_v0(act_dim, sigma0=sigma0, clip_value=clip_value)
    #
    for i_iter in range(max_iter):
        rewards = np.zeros(n_samples)
        Actions = np.zeros(shape=(n_samples, pi.act_dim))
        for j_sample in range(n_samples):
            act = pi.sample()
            #
            rewards[j_sample], _ = env.episode(act)

            #
            Actions[j_sample, :] = act
        
        # compute weights 
        beta = beta0 / (np.max(rewards) - np.min(rewards))
        weights = np.exp( beta * (rewards - np.max(rewards)))
        Weights = weights / np.mean(weights)

        #
        logger.log("********** Iteration %i ************"%i_iter)
        logger.record_tabular("iter", i_iter)
        logger.record_tabular("rews", np.mean(rewards))
        logger.record_tabular("mu", pi.mu[0])
        logger.record_tabular("scale", pi.scale[0, 0])
        logger.dump_tabular()

        # 
        progress_path = save_dir + "/training_progress"
        if not os.path.exists(progress_path):
            os.mkdir(progress_path)
        sample_path = progress_path + "/iter_{0}.h5".format(i_iter)
        np.savez(sample_path, iter=i_iter, rew=np.mean(rewards), \
            mu=pi.mu, scale=pi.scale, sample_act=Actions, \
            sample_rew=rewards)

        #
        pi.fit(Weights, Actions)        
        pi.save_weight(save_dir, i=i_iter)


    