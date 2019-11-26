import numpy as np
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines import DDPG

def train_policy_ddpg(env,policy,policy_args,total_timesteps,verbose=0,actor_lr=.5, critic_lr=.001):
    """
    Parameters
    ----------
    env : vectorized set of EncoderWrapper of a TimeLimit wrapper of a restartable env.
    policy : ddpg policy class
    policy_args : dict of keyword arguments for policy class
    total_timesteps : int, how many timesteps to train policy (i.e. 200000)
    """
    # the noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    model = DDPG(policy, env, verbose=verbose, param_noise=param_noise, action_noise=action_noise, policy_kwargs=policy_args,
                actor_lr = actor_lr, critic_lr = critic_lr)
    model.learn(total_timesteps)
    return model

