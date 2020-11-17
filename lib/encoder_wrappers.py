import gym
import numpy as np
import torch

class EncoderWrapper(gym.ObservationWrapper):
    def __init__(self, env, enc, enc_params):
        super(EncoderWrapper, self).__init__(env)
        self.env = env
        self.enc = enc
        self.enc_params = enc_params
        
    def observation(self,obs):
        return self.enc(obs,self.enc_params)

class TorchEncoderWrapper(gym.ObservationWrapper):
    def __init__(self, env, torch_enc, proj):
        super(TorchEncoderWrapper, self).__init__(env)
        self.env = env
        self.torch_enc = torch_enc
        self.proj = proj
        o = self.observation(self.env.reset())
        self.observation_space.shape = o.shape
    
    def observation(self,obs):
        with torch.no_grad(): # we aren't doing autograd, just computation
            # the torch encoder expects a batch of inputs, so create new axis
            rep = self.torch_enc(torch.from_numpy(np.expand_dims(obs,0)).float()).numpy()[0]
        return np.dot(self.proj,rep)
        
def mlp_encoder(obs,params):
    weights, biases, proj = params
    layer = np.dot(weights[0],obs)+biases[0].flatten()
    layer = np.maximum(0,layer)

    for k in range(len(weights)-1):
        layer = np.dot(weights[k+1],layer) + biases[k+1].flatten()
        layer = np.maximum(0,layer)
    
    return np.dot(proj,layer)

class NormalizedActionWrapper(gym.ActionWrapper):
    def __init__(self,env,normalizer):
        super(NormalizedActionWrapper, self).__init__(env)
        self.normalizer=normalizer
        
    def action(self,action):
        # takes a normalized actions and returns the appropriate scaled action for the env
        return self.normalizer(action)
    
    def reverse_action(self,action):
        # maybe I don't need to implement this
        pass

def identity_encoder(obs,params):
    return obs