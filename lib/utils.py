from sklearn.decomposition import TruncatedSVD
import matplotlib.cm as cm
import numpy as np
from matplotlib import pyplot as plt

def visualize_trajectory(X):
    """
    Given a trajectory of states, visualize it's projection onto the top 2 eigendirections.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, d) -- assume ordering of rows is the order of the trajectory
    """
    svd = TruncatedSVD()
    xhat = svd.fit_transform(X)
    colors = cm.rainbow(np.linspace(0,1,X.shape[0]))
    plt.scatter(xhat[:,0],xhat[:,1],c=colors)
    plt.plot(xhat[:,0],xhat[:,1])#,c=colors)
      