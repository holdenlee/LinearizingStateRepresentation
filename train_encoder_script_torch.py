from lib.restartable_pendulum import RestartablePendulumEnv
from lib.state_rep_torch import train_encoder
import gym
import numpy as np
from matplotlib import pyplot as plt
import itertools
import sys

def main():
    
    for arg in sys.argv:
        if arg.startswith('--job='):
            job_iter = int(arg.split('--job=')[1]) - 1
    
    # specify environment information
    env = RestartablePendulumEnv()
    state_dim = 3
    act_dim = 1
    
    # specify training details to loop over
    archs = [[state_dim]+arch for arch in [[64], [64,64], [64,64,64], [128], [128, 128], [128,128,128], [300], [300,300]]]
    traj_lens = [2,5,10,20]
    param_lists = [archs, traj_lens]
    
    
    i = job_iter
    tup = list(itertools.product(*param_lists))[job_iter]
    
    parameters = {
        "n_episodes" :5000,
        "batch_size" : 50,
        "learning_rate" : 1e-3,
        "widths" : tup[0],
        "traj_len" : tup[1]
    }

    widths = parameters["widths"]
    traj_len = parameters["traj_len"]
    save_dir = "./experiments/deep_dive/model_43_dive/{}".format(i)
    n_episodes = parameters["n_episodes"]
    batch_size = parameters["batch_size"]
    learning_rate = parameters["learning_rate"]    

    params, losses = train_encoder(env, traj_len, state_dim, act_dim, widths, n_episodes,
                                   lr=learning_rate,
                                   batch_size = batch_size,
                                   show_progress=False,
                                   track_loss_every = 10,
                                   drift=True
                                  )

    weights = [w for w in params[:2*(len(widths)-1):2]]
    biases = [b.flatten() for b in params[1:2*(len(widths)-1):2]]
    projectors = params[2*(len(widths)-1):2*(len(widths)-1)+traj_len+1]
                            
    
    # save the representation weights
    np.savez(save_dir + "projectors.npz",*projectors)
    np.savez(save_dir + "weights.npz",*weights)
    np.savez(save_dir + "biases.npz",*biases)
    
    # save the training params
    with open(save_dir + "train_params.txt","w") as f:
        for tup in parameters.items():
            f.write(" ".join([str(v) for v in tup]))
            f.write("\n")


    plt.plot(losses)
    plt.savefig(save_dir + "losses.png")
    plt.clf()
        

if __name__ == '__main__':
    
    main()