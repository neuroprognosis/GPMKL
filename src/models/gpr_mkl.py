"""
 Author: Nemali Aditya <aditya.nemali@dzne.de>
==================
Multikernel Learning - Gaussian process Regression - Implementation based on
C. E. Rasmussen & C. K. I. Williams, Gaussian Processes.

For implementation, refer to examples folder.
==================
"""

import numpy as np
import copy
from scipy.optimize import minimize
import pandas as pd
from numpy.linalg import inv
from preprocessing.kernel_preprocessing import cholesky_factorise
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


class GaussianProcessRegression():

    """Gaussian process Regression Multi-kernel learning (GPR-MKL).
        The implementation is of the algorithm is based on Chapter 2 of C. E. Rasmussen & C. K. I. Williams, Gaussian
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
    def __init__(self, kernel, data, target, noise, optimization=True):
        self.kernel = kernel
        self.data = data
        self.target = target
        self.noise = noise
        self.optimization = optimization

    def fit(self, train_index, *args):

        """Fit MKL version of Gaussian process regression model.
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
                    data_kernel += self.kernel(train, train)
                    loc += 1
                else:
                    raise Exception('Optimizer Failed - Reinitialize hyper-parameters')
            K = data_kernel[np.ix_(self.train_index, train_index)]
            self.K_train = K
            self.data_kernel = data_kernel
            self.hyper_parameters = hyper_parameters

        else:
            raise Exception('Input training data as List eg., fit([X], y)')
        return self

    def predict(self, test_index, sigma=False):

        """Predict using the MKL-Gaussian process classification model.
              ----------
               test_index : index of the test/validation dataset.
                sigma: If true returns covariance of the test dataset
               Returns
               -------
               mean_test : ndarray of shape (n_samples,) or (n_samples, n_targets)
                   Mean of predictive distribution a query points.
               covariance
        """

        K_train_test = self.data_kernel[np.ix_(self.train_index, test_index)]
        K_inv = inv(self.K_train)
        mean = K_train_test.T.dot(K_inv).dot(self.y_train)
        self.mean = mean
        if (sigma):
            K_test = self.data_kernel[np.ix_(self.test_index, self.test_index)]
            K_test = K_test + self.noise ** 2 * np.eye(len(K_test))
            cov = K_test - K_train_test.T.dot(K_inv).dot(K_train_test)
            self.cov = cov
        else:
            cov = False
            self.cov = cov
        return self.mean, self.cov

    def score(self, validation_target):
        '''
        predicted_mean:  Predicted mean from predict() function
        validation_target:  True target of test dataset

        Returns
        __________

        corr: Pearson's correlation
        r_score = R2_score

        '''

        self.predict_mean = self.mean
        self.validation_target = validation_target
        r_score = r2_score(self.validation_target, self.predict_mean)
        corr = pearsonr(self.validation_target, self.predict_mean)[0]
        return corr, r_score

    def optimize(self):

        """Objective function to optimize  -MKL-Gaussian process regression model.
                     Returns
                     -------
                    hyperaparameter : Estimated value of the hyperparameter
        """
        res = minimize(self.log_marginal_likelihood, self.kernel.theta(), method='L-BFGS-B', bounds=self.kernel.bounds())
        self.log_marginal_likelihood_value = -res['fun']
        self.kernel.set_theta(res['x'])
        return res['x']

    def weight_map(self, weights_data, test_target):
        """Weight Map  -MKL-Gaussian process regression model.
            Returns
            -------
            weights : Weights contribution of each voxel to predict the target
        """
        weight_cov = self.noise ** 2 * np.eye(weights_data.shape[0], weights_data.shape[0])
        weight_cov_inv = inv(weight_cov)
        alpha = weight_cov_inv.dot(test_target)
        weights = (weights_data.T) * alpha
        return weights

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
        K = kernel(self.X_train)
        K[np.diag_indices_from(K)] += self.noise
        L = cholesky_factorise(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_train))
        # Compute log-likelihood
        log_likelihood = -0.5 * np.dot(self.y_train.T, alpha)
        log_likelihood -= np.log(np.diag(L)).sum()
        log_likelihood -= K.shape[0] / 2 * np.log(2 * np.pi)
        return -log_likelihood
