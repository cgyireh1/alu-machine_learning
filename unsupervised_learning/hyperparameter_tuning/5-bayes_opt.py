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
        * f is the black-box function to be optimized
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
        Optimize method
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