import numpy as np
import tensorflow as tf
import gym
from matplotlib import pyplot as plt

def weight_variable(shape, var_name, distribution='tn', scale=0.1, first_guess=0):
    """Create a variable for a weight matrix.
    Arguments:
        shape -- array giving shape of output weight variable
        var_name -- string naming weight variable
        distribution -- string for which distribution to use for random initialization (default 'tn')
        scale -- (for tn distribution): standard deviation of normal distribution before truncation (default 0.1)
        first_guess -- (for tn distribution): array of first guess for weight matrix, added to tn dist. (default 0)
    Returns:
        a TensorFlow variable for a weight matrix
    Side effects:
        None
    Raises ValueError if distribution is filename but shape of data in file does not match input shape
    """
    if distribution == 'tn':
        initial = tf.truncated_normal(shape, stddev=scale, dtype=tf.float64) + first_guess
    elif distribution == 'xavier':
        scale = 4 * np.sqrt(6.0 / (shape[0] + shape[1]))
        initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float64)
    elif distribution == 'dl':
        # see page 295 of Goodfellow et al's DL book
        # divide by sqrt of m, where m is number of inputs
        scale = 1.0 / np.sqrt(shape[0])
        initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float64)
    elif distribution == 'he':
        # from He, et al. ICCV 2015 (referenced in Andrew Ng's class)
        # divide by m, where m is number of inputs
        scale = np.sqrt(2.0 / shape[0])
        initial = tf.random_normal(shape, mean=0, stddev=scale, dtype=tf.float64)
    elif distribution == 'glorot_bengio':
        # see page 295 of Goodfellow et al's DL book
        scale = np.sqrt(6.0 / (shape[0] + shape[1]))
        initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float64)
    else:
        initial = np.loadtxt(distribution, delimiter=',', dtype=np.float64)
        if (initial.shape[0] != shape[0]) or (initial.shape[1] != shape[1]):
            raise ValueError(
                'Initialization for %s is not correct shape. Expecting (%d,%d), but find (%d,%d) in %s.' % (
                    var_name, shape[0], shape[1], initial.shape[0], initial.shape[1], distribution))
    return tf.Variable(initial, name=var_name)


def bias_variable(shape, var_name, distribution=''):
    """Create a variable for a bias vector.
    Arguments:
        shape -- array giving shape of output bias variable
        var_name -- string naming bias variable
        distribution -- string for which distribution to use for random initialization (file name) (default '')
    Returns:
        a TensorFlow variable for a bias vector
    Side effects:
        None
    """
    if distribution:
        initial = np.genfromtxt(distribution, delimiter=',', dtype=np.float64)
    else:
        initial = tf.constant(0.0, shape=shape, dtype=tf.float64)
    return tf.Variable(initial, name=var_name)


def encoder(widths, dist_weights, dist_biases, scale, first_guess):
    """Create an encoder network: an input placeholder x, dictionary of weights, and dictionary of biases.
    Arguments:
        widths -- array or list of widths for layers of network
        dist_weights -- array or list of strings for distributions of weight matrices
        dist_biases -- array or list of strings for distributions of bias vectors
        scale -- (for tn distribution of weight matrices): standard deviation of normal distribution before truncation
        num_shifts_max -- number of shifts (time steps) that losses will use (max of num_shifts and num_shifts_middle)
        first_guess -- (for tn dist. of weight matrices): array of first guess for weight matrix, added to tn dist.
    Returns:
        x -- placeholder for input
        weights -- dictionary of weights
        biases -- dictionary of biases
    Side effects:
        None
    """
    weights = []
    biases = []

    for i in np.arange(len(widths) - 1):
        weights.append(weight_variable([widths[i+1], widths[i]], var_name='WE%d' % (i + 1),
                                                    distribution=dist_weights[i], scale=scale,
                                                    first_guess=first_guess))
        # TODO: first guess for biases too (and different ones for different weights)
        biases.append(bias_variable([widths[i+1],1], var_name='bE%d' % (i + 1),
                                                 distribution=dist_biases[i]))
    return weights, biases

def projectors(shape,num_projectors,dist,scale,first_guess):
    projectors = []
    for i in range(num_projectors):
        projectors.append(weight_variable(shape,var_name="P{}".format(i+1),distribution=dist,
                                         scale=scale,first_guess=first_guess))
    return projectors

def get_trajectory(env,start_state,action,traj_len, traj_type = "general", drift_action=[0]):
    """
    Parameters
    ----------
    env : openai gym environment
    start_state : ndarray of proper shape to specify the starting state of the environment
    action : ndarray of shape appropriate for environment action (or list of such arrays)
    traj_len : integer, specifies how many steps to collect
    traj_type : string, optional
        Specifies how to generate the trajectory
        "initial" -- take the action initially, then let drift
        "drive" -- take the same action each time
        "general" -- action in this case is a list of actions, take each one per timestep
        "initial_terminal" -- action in this case is two actions, take the first initially, 
            then let drift and finally take the second at the last step
    
    Returns
    -------
    X : ndarray of shape (traj_len, obs_dim)
        The observations from the resulting trajectory
    D : ndarray of shape (traj_len, obs_dim) (optionally)
        The corresponding drift trajectory
    """
    if traj_type in ["initial", "drive"]:
        drive = traj_type=="drive"
        obs_dim = len(env.reset(state = start_state))
        X = np.empty((traj_len, obs_dim))
        D = np.empty((traj_len, obs_dim))

        # collect trajectory with action
        obs, rew, done, _ = env.step(action)
        X[0] = obs
        for j in range(traj_len-1):
            if drive:
                obs, rew, done, _ = env.step(action)
            else:
                obs, rew, done, _ = env.step(drift_action)
            X[j+1] = obs

        # trajectory with drift
        env.reset(start_state)
        for i in range(traj_len):
            obs, rew, done, _  = env.step(drift_action)
            D[i] = obs

        return X, D
    elif traj_type == "general": #TODO: reset it in the intended start state
        obs_dim = len(env.reset(state = start_state))
        X = np.empty((traj_len+1, obs_dim))
        X[0] = start_state
        # collect trajectory with action
        obs, rew, done, _ = env.step(action[0])
        X[1] = obs
        for j in range(traj_len-1):
            obs, rew, done, _ = env.step(action[j+1])
            X[j+2] = obs
        return X
    elif traj_type == "initial_terminal":
        obs_dim = len(env.reset(state = start_state))
        X = np.empty((traj_len+1, obs_dim))
        X[0] = start_state
        # collect trajectory with action
        obs, rew, done, _ = env.step(action[0])
        X[1] = obs
        for j in range(traj_len-1):
            if j < traj_len-1:
                obs, rew, done, _ = env.step(drift_action)
            else:
                obs, rew, done, _ = env.step(action[-1])
            X[j+2] = obs
        return X

def trajectory_loss(X_enc,D_enc,action,P):
    """
    Return the loss for an encoded trajectory.
    
    Parameters
    ----------
    X : shape (traj_len, enc_dim)
        The observations from the resulting trajectory
    D : shape (traj_len, enc_dim)
        The corresponding drift trajectory
    action : shape (act_dim,)
    P : list of length traj_len, each element of shape (act_dim,enc_dim)
    
    Returns
    -------
    average coordinate-wise squared error prediction loss
    """
    losses = []
    for i in range(len(P)): # predict the action
        pred_action = tf.matmul(P[i],tf.expand_dims(X_enc[i]-D_enc[i],-1))
        err = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(pred_action,action)))
        losses.append(err)
    return sum(losses)/len(P) 

def general_trajectory_loss(X_enc,actions,P,T):
    """
    Return the loss for an encoded trajectory.
    
    Parameters
    ----------
    X : shape (traj_len+1, enc_dim)
        The observations from the resulting trajectory
    actions : shape (traj_len,act_dim)
    P : list of length traj_len+1, each element of shape (act_dim,enc_dim),
        The last element is the fixed Projector applied to the observed state
    T : list of length traj_len-1, each element of shape (act_dim,act_dim)
    
    Returns
    -------
    average coordinate-wise squared error prediction loss
    """
    losses = []
    for i in range(len(P)-1): # predict the action
        proj_xi = tf.matmul(P[-1],tf.expand_dims(X_enc[i+1],-1))
        proj_x0 = tf.matmul(P[i],tf.expand_dims(X_enc[0],-1))
        proj_acts = sum([tf.matmul(T[i-j-1],tf.expand_dims(actions[j],-1)) for j in range(i-1)])
        pred_action = proj_xi-proj_x0-proj_acts
        err = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(pred_action,action[i])))
        losses.append(err)
    return sum(losses)/len(P) 


def encode_block(block,weights,biases):
    """
    Apply the neural network encoder to a block of data columnwise.
    Weights is list of  matrices with compatible dimensions, and first entry 
    must be compatible with block.
    """
    layer = tf.matmul(weights[0],block) + biases[0]
    layer = tf.nn.relu(layer)
    for k in range(len(weights)-1):
        layer = tf.matmul(weights[k+1],layer) + biases[k+1]
        layer = tf.nn.relu(layer)
    return layer 

def train_encoder(env,start_states, start_actions, traj_len, n_passes, 
                  state_dim, act_dim,widths,
                  learning_rate=1e-3,
                  traj_type="initial",
                  init_projectors=None,
                  init_weights=None,
                  init_biases=None,
                  batch_size = 100,
                  save_dir="",
                  show_progress=True,
                  track_loss_every=1):
    """
    Use tensorflow to train the parameters of an encoder by minimizing trajectory loss.
    """
    
    
    n_episodes = len(start_states)
    
    # tensorflow boilerplate
    config = tf.ConfigProto(
        allow_soft_placement=True,
    )
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    # initialize variables, which are the network weights and biases as well as projectors 
    widths = [state_dim]+widths
    dist_weights=["dl"]*(len(widths)-1)
    scale=None
    first_guess=None
    dist_biases=[None]*(len(widths)-1)
    
    if init_projectors is None:
        projector_vars = projectors((act_dim,widths[-1]),traj_len,"dl",scale,first_guess)
    else:
        projector_vars = [tf.Variable(tf.convert_to_tensor(proj),dtype=tf.float64) for proj in init_projectors]
        
    if init_weights is None:
        weight_vars, bias_vars = encoder(widths, dist_weights, dist_biases, scale, first_guess)
    else:
        weight_vars = [tf.Variable(tf.convert_to_tensor(wt),dtype=tf.float64) for wt in init_weights]
        bias_vars = [tf.Variable(tf.convert_to_tensor(bs),dtype=tf.float64) for bs in init_biases]
        
    
    
    with sess.as_default():
        # train on batch of trajectories at a time
        # placeholder for data from a batch of trajectories
        input_trajectory_batch = [tf.placeholder(tf.float64, shape=[traj_len,state_dim]) for _ in range(batch_size)]
        input_act_batch = [tf.placeholder(tf.float64, shape=[act_dim]) for _ in range(batch_size)]
        input_drift_batch = [tf.placeholder(tf.float64, shape=[traj_len,state_dim]) for _ in range(batch_size)]
        
        # encode the batch of trajectories and drifts
        X_enc = [tf.transpose(encode_block(tf.transpose(b),weight_vars,bias_vars)) for b in input_trajectory_batch]
        D_enc = [tf.transpose(encode_block(tf.transpose(b),weight_vars,bias_vars)) for b in input_drift_batch]
        
        # create the batch loss
        loss = sum([trajectory_loss(x_enc,d_enc,act,projector_vars) for x_enc,d_enc,act in zip(X_enc,D_enc,input_act_batch)])/batch_size
        
        # create the minimizer
        global_step = tf.Variable(0,name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) # traditionally have chosen learning_rate=1e-3
        train_op = optimizer.minimize(loss, global_step)
            
        sess.run(tf.global_variables_initializer())
        
        losses = []
        
        track_loss_every = max(1,track_loss_every)
         
        for p in range(n_passes):
       
            if show_progress:
                print("\nEpoch {}\n".format(p))
            
            samples = np.random.permutation(n_episodes)
            avg_loss = 0
            for j in range(int(n_episodes/batch_size)):
                if (j%track_loss_every)==0:
                    losses.append(avg_loss/track_loss_every)
                    avg_loss = 0
                
                batch_inds = samples[j:j+batch_size]
                actions = [start_actions[ind] for ind in batch_inds]
                eps = [get_trajectory(env,start_states[ind],start_actions[ind],traj_len,traj_type=traj_type)
                      for ind in batch_inds]
                
                feed_dict = {d : i for d, i in zip(input_act_batch, actions)}
                feed_dict.update({d : i[0] for d, i in zip(input_trajectory_batch,eps)})
                feed_dict.update({d : i[1] for d, i in zip(input_drift_batch,eps)})
      
                _, loss_val, step = sess.run([train_op, loss, global_step], feed_dict=feed_dict)
                #losses.append(loss_val)
                avg_loss += loss_val
                if show_progress:
                    print("Epoch Completion: {0:.3f}%, Loss: {1}".format(100*j*batch_size/n_episodes,loss_val),end="\r",flush=True)
            if (j%track_loss_every) > 0:
                losses.append(avg_loss/(j%track_loss_every + 1))
    return [v.eval(sess) for v in projector_vars], [v.eval(sess) for v in weight_vars], [v.eval(sess) for v in bias_vars], losses[1:]

def train_general_encoder(env,start_states, action_seqs, traj_len, n_passes, 
                  state_dim, act_dim,widths,
                  learning_rate=1e-3,
                  traj_type="general",
                  init_projectors=None, 
                  init_T=None,
                  init_weights=None,
                  init_biases=None,
                  batch_size = 100,
                  save_dir="",
                  show_progress=True,
                  track_loss_every=1):
    """
    Use tensorflow to train the parameters of an encoder by minimizing trajectory loss.
    """
    
    
    n_episodes = len(start_states)
    
    # tensorflow boilerplate
    config = tf.ConfigProto(
        allow_soft_placement=True,
    )
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    # initialize variables, which are the network weights and biases as well as projectors 
    widths = [state_dim]+widths
    dist_weights=["dl"]*(len(widths)-1)
    scale=None
    first_guess=None
    dist_biases=[None]*(len(widths)-1)
    
    if init_P is None:
        P_vars = projectors((act_dim,widths[-1]),traj_len+1,"dl",scale,first_guess)
    else:
        P_vars = [tf.Variable(tf.convert_to_tensor(proj),dtype=tf.float64) for proj in init_P]
    
    if init_T is None:
        T_vars = projectors((act_dim,act_dim), traj_len-1, "dl", scale, first_guess)
    else:
        T_vars = [tf.Variable(tf.convert_to_tensor(T),dtype=tf.float64) for T in init_T]
    
    if init_weights is None:
        weight_vars, bias_vars = encoder(widths, dist_weights, dist_biases, scale, first_guess)
    else:
        weight_vars = [tf.Variable(tf.convert_to_tensor(wt),dtype=tf.float64) for wt in init_weights]
        bias_vars = [tf.Variable(tf.convert_to_tensor(bs),dtype=tf.float64) for bs in init_biases]
        
    
    with sess.as_default():
        # train on batch of trajectories at a time
        # placeholder for data from a batch of trajectories
        input_trajectory_batch = [tf.placeholder(tf.float64, shape=[traj_len+1,state_dim]) for _ in range(batch_size)]
        input_act_batch = [tf.placeholder(tf.float64, shape=[traj_len,act_dim]) for _ in range(batch_size)]
        
        # encode the batch of trajectories and drifts
        X_enc = [tf.transpose(encode_block(tf.transpose(b),weight_vars,bias_vars)) for b in input_trajectory_batch]
        
        # create the batch loss
        loss = sum([general_trajectory_loss(x_enc,act,P_vars,T_vars) for x_enc,act in zip(X_enc,input_act_batch)])/batch_size
        
        # create the minimizer
        global_step = tf.Variable(0,name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) # traditionally have chosen learning_rate=1e-3
        train_op = optimizer.minimize(loss, global_step)
            
        sess.run(tf.global_variables_initializer())
        
        losses = []
        
        track_loss_every = max(1,track_loss_every)
         
        for p in range(n_passes):
       
            if show_progress:
                print("\nEpoch {}\n".format(p))
            
            samples = np.random.permutation(n_episodes)
            avg_loss = 0
            for j in range(int(n_episodes/batch_size)):
                if (j%track_loss_every)==0:
                    losses.append(avg_loss/track_loss_every)
                    avg_loss = 0
                
                batch_inds = samples[j:j+batch_size]
                actions = [action_seqs[ind] for ind in batch_inds]
                initstates = [start_states[ind] for ind in batch_inds]
                eps = [get_trajectory(env,start_states[ind],action_seqs[ind],traj_len,traj_type=traj_type)
                      for ind in batch_inds]
                
                feed_dict = {d : i for d, i in zip(input_act_batch, actions)}
                feed_dict.update({d : i for d, i in zip(input_trajectory_batch,eps)})
                
                _, loss_val, step = sess.run([train_op, loss, global_step], feed_dict=feed_dict)
                #losses.append(loss_val)
                avg_loss += loss_val
                if show_progress:
                    print("Epoch Completion: {0:.3f}%, Loss: {1}".format(100*j*batch_size/n_episodes,loss_val),end="\r",flush=True)
            if (j%track_loss_every) > 0:
                losses.append(avg_loss/(j%track_loss_every + 1))
    return [v.eval(sess) for v in projector_vars], [v.eval(sess) for v in weight_vars], [v.eval(sess) for v in bias_vars], losses[1:]

