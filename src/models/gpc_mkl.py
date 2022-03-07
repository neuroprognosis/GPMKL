"""
 Author: Nemali Aditya <aditya.nemali@dzne.de>
==================
Multikernel Learning - Gaussian process Classification - Implementation based on
C. E. Rasmussen & C. K. I. Williams, Gaussian Processes.

For implementation, refer to examples folder.
==================
"""


import numpy as np
import copy
from scipy.optimize import minimize
from scipy.stats import norm
import pandas as pd
from numpy.linalg import inv

class GaussianProcessClassifier():

    """Gaussian process Classification Multi-kernel learning (GPC-MKL).
        The implementation is of the algorithm is based on Chapter 3 of C. E. Rasmussen & C. K. I. Williams, Gaussian
        Processes Book.
        ----------
        kernel : kernel instance, default=None
            The kernel specifying the covariance function of the GP. Note that
            the kernel hyperparameters are optimized during fitting unless the
            optimization are marked as "False".

        data : List of pandas dataframe object (full dataset - including test & train)
        target : pandas dataframe of shape (n_samples,) or (n_samples, n_targets)
            Target values for the full dataset (also required for prediction).

        optimization : Per default, set to True for kernel hyperparameters optimization.

    """
    def __init__(self, kernel, data, target, optimization=True):
        self.kernel = kernel
        self.data = data
        self.target = target
        self.optimization = optimization
    def fit(self, train_index, *args):

        """Fit MKL version of Gaussian process classification model.
                Parameters
                ----------
                train_index : index of the training data.

                Returns
                -------
                self : object
                    GaussianProcessClassifier class instance.
                """
        self.train_index = train_index
        if isinstance(self.target, list):
            self.y_train = self.target[0]
            self.y_train = pd.DataFrame(self.target, index=train_index)
        else:
            self.y_train = pd.DataFrame(self.target, index=train_index)
            self.y_train = np.array(self.y_train).ravel()

        # Estimating kernels for each data object
        K = 0
        data_kernel = 0
        hyper_parameters = []
        if isinstance(self.data, list):
            loc = 0
            for train in self.data:
                self.X_train =  pd.DataFrame(train, index=train_index)
                self.X_train = np.array(self.X_train)
                # Optimization of kernels
                if (self.optimization):
                    result = self.optimize()
                    hyper_parameters.append(result[0])
                    if(result):
                        optim_true = True
                    else:
                        raise Exception('Optimizer Failed - Reinitialize hyper-parameters')
                else:
                    self.kernel.set_theta(args[0][loc])
                    optim_true = True
                if (optim_true):
                    K += self.kernel(self.X_train, self.X_train)
                    data_kernel += self.kernel(train, train)
                    loc += 1
                else:
                    raise Exception('Optimizer Failed - Reinitialize hyper-parameters')

            mode = self.mode(K, iterations=500)
            self.data_kernel = data_kernel
            self.hyper_parameters = hyper_parameters

        else:
            raise Exception('Input training data as List eg., fit([X], y)')
        return self

    def predict(self, test_index):

        """Predict using the MKL-Gaussian process classification model.
              ----------
               test_index : index of the test/validation dataset.

               Returns
               -------
               mean_test : ndarray of shape (n_samples,) or (n_samples, n_targets)
                   Mean of predictive distribution a query points.
               predicted target: np.where(mean_test > 0, 1, -1)
        """
        K_train_test = self.data_kernel[np.ix_(self.train_index, test_index)]
        mean_test = K_train_test.T.dot(self.N * self.y_train / self.phi)
        return mean_test, np.where(mean_test > 0, 1, -1)

    def optimize(self):

        """Objective function to optimize  -MKL-Gaussian process classification model.
                     Returns
                     -------
                    hyperaparameter : Estimated value of the hyperparameter
        """
        res = minimize(self.log_marginal_likelihood, self.kernel.theta(), method='L-BFGS-B', bounds=self.kernel.bounds())
        self.log_marginal_likelihood_value = -res['fun']
        self.kernel.set_theta(res['x'])
        return res['x']

    def predict_probablity(self, test_index):
        """Predict probability using the MKL-Gaussian process classification model.
            ----------
            test_index : index of the test/validation dataset.

            Returns
            -------
            probabilty : ndarray of shape (n_samples,) or (n_samples, n_targets)
        """

        K_train_test = self.data_kernel[np.ix_(self.train_index, test_index)]
        K_test = self.data_kernel[np.ix_(test_index, test_index)]
        mean_test = K_train_test.T.dot(self.N * self.y_train / self.phi)
        cov = np.linalg.solve(self.L, self.W_square[:, np.newaxis] * K_train_test)
        cov_test = np.diag(K_test) - np.einsum("ij,ij->j", cov, cov)
        prob_test = norm.cdf(mean_test / np.sqrt(1 + cov_test))
        return prob_test

    def weight_map(self, weights_data, test_target):
        """Weight Map  -MKL-Gaussian process classification model.
            Returns
            -------
            weights : Weights contribution of each voxel to predict the target
        """
        temp = 0.3 * np.eye(weights_data.shape[0], weights_data.shape[0])
        temp_inv = inv(temp)
        test_target = np.array(test_target)
        alpha = temp_inv.dot(test_target)
        weights = (weights_data.T) * alpha
        return weights

    def mode(self, K, iterations=200):
        '''
            For implementation of Algorithm: see page 46 C. E. Rasmussen & C. K. I. Williams, Gaussian Processes
            for Machine Learning
            K: covariance matrix
            y_train: y (±1 targets)
        '''
        f = np.zeros_like(self.y_train, dtype=np.float64)
        # Newton method
        log_marginal_likelihood = -np.inf
        for iter in range(iterations):
            # estimating W using eq 3.16
            phi = norm.cdf(self.y_train * f)
            N = norm.pdf(f)
            W = N ** 2 / phi ** 2 + self.y_train * f * N / phi
            W_square = np.sqrt(W)
            W_square_cov = W_square[:, np.newaxis] * K
            B = np.eye(W.shape[0]) + W_square_cov * W_square[np.newaxis, :]
            L = np.linalg.cholesky(B)
            b = W * f + N * self.y_train / phi
            a = b - W_square * np.linalg.solve(L.T, np.linalg.solve(L, W_square_cov.dot(b)))
            f = K.dot(a)

            approx_log_marginal_likelihood = -0.5 * a.T.dot(f) + np.log(phi).sum() - np.log(np.diag(L)).sum()
            if approx_log_marginal_likelihood - log_marginal_likelihood < 1e-10:
                break
            log_marginal_likelihood = approx_log_marginal_likelihood
        self.f, self.phi, self.N, self.W_square, self.L = f, phi, N, W_square, L
        return log_marginal_likelihood

    def log_marginal_likelihood(self, theta):
        """Return log-marginal likelihood for the training data.
               Parameters
               ----------
               theta : array-like of shape (n_kernel_params,) default=None
                   Kernel hyperparameters for which the log-marginal likelihood is
                   evaluated. If None, the precomputed log_marginal_likelihood
                   of ``self.kernel_.theta`` is returned.

               Returns
               -------
               log_likelihood : float
                   Log-marginal likelihood of theta for training data.
               """
        kernel = copy.deepcopy(self.kernel)
        kernel.set_theta(theta)
        K_train = kernel(self.X_train)
        mode = self.mode(K_train, iterations=500)
        return -mode
