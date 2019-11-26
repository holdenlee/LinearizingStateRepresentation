import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
import gym

# format a batch as a list of states (each entry of list is shape (batch_size, state_size), corresponding to x0, x1,...)
# and list of actions (each entry of shape (batch_size, action_size), u0,u1,...)

def get_trajectory_batch(env,T,batch_size):
    X_list = [np.empty((batch_size,env.observation_space.shape[0])) for _ in range(T+1)]
    U_list = [4*np.random.rand(batch_size,env.action_space.shape[0])-2 for _ in range(T)]
    for j in range(batch_size):
        X_list[0][j,:] = env.reset()
        for i in range(T):
            X_list[i+1][j,:] = env.step(U_list[i][j])[0]
    return [torch.from_numpy(X).float() for X in X_list], [torch.from_numpy(U).float() for U in U_list]

def get_trajectory_batch_drift(env,T,batch_size):
    X_list = [np.empty((batch_size,env.observation_space.shape[0])) for _ in range(T+1)]
    X_drift_list = [np.empty((batch_size,2)) for _ in range(T+1)] # hardcoded
    U_list = [4*np.random.rand(batch_size,env.action_space.shape[0])-2 for _ in range(2)]
    for j in range(batch_size):
        X_list[0][j,:] = env.reset()
        X_drift_list[0][j,:] = env._get_state()
        X_list[1][j,:] = env.step(U_list[0][j])[0] # always take this first action
        X_drift_list[1][j,:] = env._get_state()
        for i in range(T-1):
            X_list[i+2][j,:] = env.step(U_list[1][j])[0]
            env.reset(state = X_drift_list[i+1][j,:]) # reset to the  drift state previous
            env.step([0])
            X_drift_list[i+2][j,:] = env._get_state() # drift to get the new drift state
            
    return [torch.from_numpy(X).float() for X in X_list], [torch.from_numpy(U).float() for U in U_list]

class EncoderNet(nn.Module):
    def __init__(self, widths, concat_input=False):
        super(EncoderNet, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(widths[i],widths[i+1]) for i in range(len(widths)-1)])
        self.concat_input = concat_input
        
    def forward(self, x):
        z = x
        for layer in self.layers:
            z = F.relu(layer(z))
        if self.concat_input:
            z = torch.cat((z,x),dim=1)
        return z

class PredictorNet(nn.Module):
    def __init__(self,encoder,T, enc_dim, act_dim):
        super(PredictorNet, self).__init__()
        self.encoder = encoder
        self.T = T # length of the trajectory
        self.state_projectors = nn.ModuleList([nn.Linear(enc_dim,act_dim,bias=False) for i in range(T+1)])
        self.action_projectors = nn.ModuleList([nn.Linear(act_dim,act_dim,bias=False) for i in range(T-1)])
    
    def compute_loss(self, X_list, U_list):
        """
        Parameters
        ----------
        X_list : length T+1 list of tensors, each of shape (N,state_dim) (N is batch size)
            The ith tensor in the list gives the batch of ith states in the trajectories
        U_list : length T list of tensors, each of shape (N,act_dim)
            The ith tensor in the list gives the batch of ith actions in the trajectories
            
        Returns
        -------
        loss : scalar, the average squared error (over all coordinates of the batch, and over the trajectory)
        """
        enc_X_list = [self.encoder(X) for X in X_list]
        loss = 0
        for i in range(self.T):
            pred = self.state_projectors[-1](enc_X_list[i+1])
            pred -= self.state_projectors[i](enc_X_list[0])
            for j in range(i-1):
                pred -= self.action_projectors[i-j-1](U_list[j])
            loss += ((pred - U_list[i])**2).mean()
        return loss/self.T
    
    def compute_loss_drift(self, X_list, U_list):
        """
        Parameters
        ----------
        X_list : length T+1 list of tensors, each of shape (N,state_dim) (N is batch size)
            The ith tensor in the list gives the batch of ith states in the trajectories
        U_list : length 2 list of tensors, each of shape (N,act_dim)
            The ith tensor in the list gives the batch of ith actions in the trajectories
            
        Returns
        -------
        loss : scalar, the average squared error (over all coordinates of the batch, and over the trajectory)
        """
        enc_X_list = [self.encoder(X) for X in X_list]
        loss = 0
        for i in range(self.T):
            pred = self.state_projectors[-1](enc_X_list[i+1])
            pred -= self.state_projectors[i](enc_X_list[0])
            pred -= self.action_projectors[i-1](U_list[0])

            loss += ((pred - U_list[1])**2).mean()
        return loss/self.T 
    
def train_encoder(env, # gym-style environment, but can reset state in specific configuration
                  T, # length of trajectory
                  state_dim, 
                  act_dim,
                  enc_arch,
                  n_episodes, # how many batches of trajectories to train on
                  lr=1e-3, #learning rate for optimizer
                  batch_size = 50,
                  show_progress=True,
                  track_loss_every=10, # print a progress statement every _ batches
                  drift=False,
                 ):
    

    enc_dim = enc_arch[-1]
    
    encNet = EncoderNet(enc_arch,concat_input=False)
    predNet = PredictorNet(encNet,T,enc_dim,act_dim)

    optimizer = optim.Adam(predNet.parameters())
    losses = []
    running_loss = 0.0
    for i in range(n_episodes):
        # generate a batch of trajectories
        if not drift:
            X_batch, U_batch = get_trajectory_batch(env,T,batch_size)
        else:
            X_batch, U_batch = get_trajectory_batch_drift(env,T,batch_size)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        if not drift:
            loss = predNet.compute_loss(X_batch, U_batch)
        else:
            loss = predNet.compute_loss_drift(X_batch, U_batch)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if ((i+1)%track_loss_every) == 0:
            if show_progress:
                print("Epoch Completion: {0:.3f}%, Loss: {1}".format(100*(i+1)/n_episodes, running_loss/track_loss_every),
                      end="\r",flush=True)
            losses.append(running_loss/track_loss_every)
            running_loss = 0.0
            
    return [p.data.numpy() for p in list(predNet.parameters())],  losses