import numpy as np
from preprocessing.kernel_preprocessing import kernel_centering, kernel_normalization
from scipy.spatial.distance import pdist, cdist, squareform


class Linear():
    def __init__(self, signal_variance=1.0,  length_scale_bounds=(1e-5, 10), kernel_scaling=True):
        self.signal_variance = signal_variance
        self.length_scale_bounds = [length_scale_bounds]
        self.kernel_scaling = kernel_scaling

    def __call__(self, X1, X2=None):
        if X2 is None:
            K = np.dot(X1, X1.T)
            if(self.kernel_scaling):
                K = kernel_centering(K)
                K = kernel_normalization(K)
            K = self.signal_variance *K 
        else:
            K = np.dot(X1, X2.T)
            if (self.kernel_scaling):
                K = kernel_centering(K)
                K = kernel_normalization(K)
            K = self.signal_variance * K
        return K

    def theta(self):
        return np.array(self.signal_variance)

    def set_theta(self, theta):
        self.signal_variance = theta
        
    def bounds(self):
        return self.length_scale_bounds


class RBF():

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5), kernel_scaling=True):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

    def __call__(self, X1, X2=None):
        if X2 is None:
            dists = pdist(X1 / self.length_scale, 'sqeuclidean')
            K = np.exp(-0.5 * dists)
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            dists = cdist(X1 / self.length_scale, X2 / self.length_scale, metric='sqeuclidean')
            K = np.exp(-0.5 * dists)
        K = kernel_centering(K)
        K = kernel_normalization(K)
        return K

    def theta(self):
        return np.array([self.length_scale])

    def set_theta(self, theta):
        [self.length_scale] = theta

    def bounds(self):
        return [self.length_scale_bounds]
    
    

    
        
