from lib.restartable_pendulum import RestartablePendulumEnv
from lib.state_rep import train_encoder
import numpy as np
from matplotlib import pyplot as plt
import itertools

def main():
    
    # specify environment information
    env = RestartablePendulumEnv()
    state_dim = 3
    act_dim = 1
    
    # specify training details to loop over
    #archs = [[64], [64,64], [64,64,64], [128], [128, 128], [128,128,128]]
    #traj_lens = [2,5,10,20]
    archs = [[128,128,128]]
    traj_lens = [10]
    param_lists = [archs, traj_lens]
    
    total_models = len(list(itertools.product(*param_lists)))
    print(total_models)
    
    for i,tup in enumerate(itertools.product(*param_lists)): # loop over the various architectures
        print("\nStarting {0} of {1} representations\n".format(i+1,total_models))
        
        parameters = {
            "n_episodes" : 3*20000,
            "n_passes" : 1,
            "batch_size" : 100,
            "learning_rate" : 1e-3,
            "widths" : tup[0],
            "traj_len" : tup[1]
        }
        
        widths = parameters["widths"]
        traj_len = parameters["traj_len"]
        save_dir = "./experiments/{}".format(i)
        n_episodes = parameters["n_episodes"]
        n_passes = parameters["n_passes"]
        batch_size = parameters["batch_size"]
        learning_rate = parameters["learning_rate"]    
        
        init_projectors=None
        init_weights=None
        init_biases=None
    
        # generate the seeds for the training trajectories
        start_states = [np.array([(np.random.rand(1)[0]*2 - 1)*np.pi, (np.random.rand(1)[0]*2 - 1)*8]) 
                        for _ in range(n_episodes)]
        start_actions = [np.random.rand(1)*4-2 for _ in range(n_episodes)]

     
        projectors,weights,biases,losses = train_encoder(env, start_states, start_actions, traj_len, n_passes, 
                                                         state_dim, act_dim, widths,
                                                         learning_rate=learning_rate,
                                                         init_projectors=init_projectors,
                                                         init_weights=init_weights,
                                                         init_biases=init_biases,
                                                         batch_size = batch_size,
                                                         save_dir = save_dir,
                                                         track_loss_every = int(n_episodes/(batch_size*100)))
    
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
        plt.show()
        plt.clf()
        

if __name__ == '__main__':
    main()