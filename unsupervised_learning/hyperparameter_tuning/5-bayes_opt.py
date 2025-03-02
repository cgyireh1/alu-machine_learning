#!/usr/bin/env python3
"""
Bayesian Optimization
"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization():
    """
    A class that performs Bayesian optimization
    on a noiseless 1D Gaussian process
    """

    def __init__(self, f, X_init, Y_init, bounds,
                 ac_samples, l=1, sigma_f=1, xsi=0.01,
                 minimize=True):
        """
        f is the black-box function to be optimized
        X_init is a numpy.ndarray of shape (t, 1) representing
          the inputs already sampled with the black-box function
        Y_init is a numpy.ndarray of shape (t, 1) representing
          the outputs of the black-box function for each input
          in X_init
          - t is the number of initial samples
        bounds is a tuple of (min, max) representing the bounds
          of the space in which to look for the optimal point
        ac_samples is the number of samples that should be analyzed
          during acquisition
        l is the length parameter for the kernel
        sigma_f is the standard deviation given to the output of
          the black-box function
        xsi is the exploration-exploitation factor for acquisition
        attributes:
          - f: the black-box function
          - gp: an instance of the class GaussianProcess
          - X_s: a numpy.ndarray of shape (ac_samples, 1) containing
            all acquisition sample points, evenly spaced between min
            and max
          - xsi: the exploration-exploitation factor
          - minimize: a bool for minimization versus maximization
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        min, max = bounds
        X_s = np.linspace(min, max, ac_samples)
        self.X_s = (np.sort(X_s)).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculates the next best sample location
        """
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize is True:
            optimize = np.amin(self.gp.Y)
            imp = optimize - mu - self.xsi

        else:
            optimize = np.amax(self.gp.Y)
            imp = mu - optimize - self.xsi

        Z = np.zeros(sigma.shape[0])

        for i in range(sigma.shape[0]):
            if sigma[i] != 0:
                Z[i] = imp[i] / sigma[i]
            else:
                Z[i] = 0

        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

        index = np.argmax(ei)
        best_sample = self.X_s[index]

        return (best_sample, ei)

    def optimize(self, iterations=100):
        """
        The optimize method
        """

        X_all_s = []
        for i in range(iterations):
            # Find the next sampling point xt by optimizing the acquisition
            # function over the GP: xt = argmaxx μ(x | D1:t−1)

            x_opt, _ = self.acquisition()
            # Optimization should stop early If
            # the next proposed point is one 
            # that has already been sampled
            if x_opt in X_all_s:
                break

            y_opt = self.f(x_opt)

            # Adding the sample to previous samples
            # D1: t = {D1: t−1, (xt, yt)} and update the GP
            self.gp.update(x_opt, y_opt)
            X_all_s.append(x_opt)

        if self.minimize is True:
            index = np.argmin(self.gp.Y)
        else:
            index = np.argmax(self.gp.Y)

        self.gp.X = self.gp.X[:-1]

        x_opt = self.gp.X[index]
        y_opt = self.gp.Y[index]

        return x_opt, y_opt
