import gym
import numpy as np
from scipy import linalg as la
import itertools
import sys

from lib.train_policy import train_policy_ddpg
from lib.linear_policies import LinearPolicy_MLPCritic
from lib.restartable_pendulum import RestartablePendulumEnv
from lib.encoder_wrappers import EncoderWrapper, mlp_encoder           

from gym.wrappers.time_limit import TimeLimit
from stable_baselines.ddpg.policies import MlpPolicy, FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DDPG

def main():
    
    # train the policy, then do some tests to get a sense of how it performs

    for arg in sys.argv:
        if arg.startswith('--job='):
            i = int(arg.split('--job=')[1]) - 1
    
    # pull in the encoder params
    p_dir = "../LinearizingRepresentations/experiments/state_rep_params/pendulum/{}".format(i)
    proj = np.load(p_dir+"projectors.npz")
    proj = np.row_stack([v for k,v in proj.items()])
    proj = la.svd(proj,full_matrices=False)[2]
    enc_dim = proj.shape[0]
    weights = np.load(p_dir+"weights.npz")
    biases = np.load(p_dir+"biases.npz")
    weights = [v for k,v in weights.items()]
    biases = [v for k,v in biases.items()]
    
    # set up the environment
    env = TimeLimit(RestartablePendulumEnv(enc_dim=enc_dim), max_episode_steps=200) # not sure effect of max_episode_steps
    env = EncoderWrapper(env,mlp_encoder,[weights,biases,proj])
    env = DummyVecEnv([lambda: env])
    pol = LinearPolicy_MLPCritic
    pol_args = dict(layers=[64, 64, 64]) # this is the architecture for the critic in ddpg, doesn't specify policy
    
    # train the model
    model = train_policy_ddpg(env,pol,pol_args,200000)
    
    # specify the path to save the model
    saveload_path = "./experiments/learned_controllers/pendulum/{}".format(i)
    model.save(saveload_path+"model")
    
    
    # clean up
    env.close()
    
    
    #model = DDPG.load(saveload_path+"model")
    
    # now let's test the model 
    # specify the test task
    n_test_steps = 100

    # uniform grid over statespace (20 points)
    angs = np.linspace(-np.pi, np.pi, 5)[:-1]
    vels = np.linspace(-1, 1, 5)
    test_states = np.array(list(itertools.product(angs,vels)))
    n_test_states = len(angs)*len(vels)
    performance = np.zeros(n_test_states)
    
    # restart the env
    env = TimeLimit(RestartablePendulumEnv(), max_episode_steps=200)
    env = EncoderWrapper(env,mlp_encoder,[weights,biases,proj])
    
    # for each test state, start the env in the state, then run forward and collect rewards
    for k in range(n_test_states):
        obs = env.reset(state=test_states[k])
        rewards = []
        for j in range(n_test_steps):
            action, _states = model.predict(obs)
            obs, reward, dones, info = env.step(action)
            rewards.append(reward)
            #env.render()
        performance[k] = np.array(rewards).mean()
    
    # clean up and save results
    env.close()
    del model
    np.savetxt(saveload_path+"test_performance.txt",performance)
    
if __name__ == '__main__':
    
    main()