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
            i = int(arg.split('--job=')[1])-1
    
    # specify environment information
    env = RestartablePendulumEnv()
    state_dim = 3
    act_dim = 1
    
    # specify training details to loop over
    jobs = [2, 5, 6, 8, 9, 10, 13, 20, 24, 28, 32]
    archs = [[state_dim]+arch for arch in [[128],
                                           [256],
                                           [512],
                                           [1024],
                                           [128,128],
                                           [256,256],
                                           [512,512],
                                           [512,256],
                                           [512,128]
                                          ]]
    traj_lens = [20]
    lrs = [.0001, .0005, .001, .005]
    param_lists = [archs, traj_lens, lrs]
    
    
    tup = list(itertools.product(*param_lists))[jobs[i]]
    
    parameters = {
        "n_episodes" :30000,
        "batch_size" : 50,
        "learning_rate" : tup[2],
        "widths" : tup[0],
        "traj_len" : tup[1]
    }

    widths = parameters["widths"]
    traj_len = parameters["traj_len"]
    save_dir = "./experiments/extra_train_exps/{}".format(i)
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