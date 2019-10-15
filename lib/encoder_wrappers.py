import gym
import numpy as np

class EncoderWrapper(gym.ObservationWrapper):
    def __init__(self, env, enc, enc_params):
        super(EncoderWrapper, self).__init__(env)
        self.env = env
        self.enc = enc
        self.enc_params = enc_params
    def observation(self,obs):
        return self.enc(obs,self.enc_params)

def mlp_encoder(obs,params):
    weights, biases, proj = params
    layer = np.dot(weights[0],obs)+biases[0].flatten()
    layer = np.maximum(0,layer)

    for k in range(len(weights)-1):
        layer = np.dot(weights[k+1],layer) + biases[k+1].flatten()
        layer = np.maximum(0,layer)
    
    return np.dot(proj,layer)
    
def identity_encoder(obs,params):
    return obs