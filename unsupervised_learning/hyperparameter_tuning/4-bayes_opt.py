#!/usr/bin/env python3
"""
Bayesian Optimization - Acquisition
"""

import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    A class that performs Bayesian optimization
    on a noiseless 1D Gaussian process
    """
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """
        init method - bayesian optimization
        """
        # black-box function
        self.f = f

        # Gaussian Process
        self.gp = GP(X_init, Y_init, l, sigma_f)

        # X_s all acquisition sample
        X_s = np.linspace(bounds[0], bounds[1], num=ac_samples)
        self.X_s = X_s.reshape(-1, 1)

        # exploration-explotation
        self.xsi = xsi

        # minimization versus maximization
        self.minimize = minimize

    def acquisition(self):
        """
        Calculates the next best sample location
        """
        # source: http://krasserm.github.io/2018/03/21/bayesian-optimization/
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize is True:
            Y_sample = np.min(self.gp.Y)
            imp = Y_sample - mu - self.xsi
        else:
            Y_sample = np.max(self.gp.Y)
            imp = mu - Y_sample - self.xsi

        Z = np.zeros(sigma.shape[0])
        for i in range(sigma.shape[0]):
            # formula if σ(x)>0 : μ(x)−f(x+)−ξ / σ(x)
            if sigma[i] > 0:
                Z[i] = imp[i] / sigma[i]
            # formula if σ(x)=0
            else:
                Z[i] = 0
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)

        X_next = self.X_s[np.argmax(ei)]

        return X_next, ei
