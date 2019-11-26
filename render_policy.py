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

i = 15
p_dir = "./experiments/state_rep_params/pendulum/{}".format(i)
proj = np.load(p_dir+"projectors.npz")
proj = np.row_stack([v for k,v in proj.items()])
proj = la.svd(proj,full_matrices=False)[2]
enc_dim = proj.shape[0]
weights = np.load(p_dir+"weights.npz")
biases = np.load(p_dir+"biases.npz")
weights = [v for k,v in weights.items()]
biases = [v for k,v in biases.items()]

saveload_path = "./experiments/learned_controllers/pendulum/{}".format(i)
model = DDPG.load(saveload_path+"model")

# now let's test the model 
# specify the test task
n_test_steps = 100

# restart the env
env = TimeLimit(RestartablePendulumEnv(), max_episode_steps=200)
env = EncoderWrapper(env,mlp_encoder,[weights,biases,proj])

# for each test state, start the env in the state, then run forward and collect rewards
for k in range(3):
    high = np.array([np.pi,1])
    start_state = np.random.uniform(low=-high,high=high) 
    obs = env.reset(state=start_state)
    for j in range(n_test_steps):
        action, _states = model.predict(obs)
        obs, reward, dones, info = env.step(action)
        env.render()
        
# clean up and save results
env.close()
del model