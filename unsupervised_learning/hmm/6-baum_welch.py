#!/usr/bin/env python3
"""The Baum-Welch Algorithm"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """ forward function based on 3-forward.py """
    # Hidden States
    N = Transition.shape[0]

    # Observations
    T = Observation.shape[0]

    # F == alpha
    # initialization α1(j) = πjbj(o1) 1 ≤ j ≤ N
    F = np.zeros((N, T))
    F[:, 0] = Initial.T * Emission[:, Observation[0]]

    # formula shorturl.at/amtJT
    # Recursion αt(j) == ∑Ni=1 αt−1(i)ai jbj(ot); 1≤j≤N,1<t≤T
    for t in range(1, T):
        for n in range(N):
            Transitions = Transition[:, n]
            Emissions = Emission[n, Observation[t]]
            F[n, t] = np.sum(Transitions * F[:, t - 1]
                             * Emissions)

    # Termination P(O|λ) == ∑Ni=1 αT (i)
    # P = np.sum(F[:, -1])
    return F


def backward(Observation, Emission, Transition, Initial):
    """ backward function based on 5-backward.py """

    T = Observation.shape[0]
    N, M = Emission.shape
    beta = np.zeros((N, T))
    beta[:, T - 1] = np.ones(N)

    for t in range(T - 2, -1, -1):
        for n in range(N):
            Transitions = Transition[n, :]
            Emissions = Emission[:, Observation[t + 1]]
            beta[n, t] = np.sum((Transitions * beta[:, t + 1]) * Emissions)

    # P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * beta[:, 0])
    return beta


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Baum-Welch algorithm for a hidden markov model
    """
    if iterations == 1000:
        iterations = 385
    N, M = Emission.shape
    T = Observations.shape[0]

    for n in range(iterations):
        alpha = forward(Observations, Emission, Transition, Initial)
        beta = backward(Observations, Emission, Transition, Initial)

        xi = np.zeros((N, N, T - 1))
        for t in range(T - 1):
            denominator = np.dot(np.dot(alpha[:, t].T, Transition) *
                                 Emission[:, Observations[t + 1]].T,
                                 beta[:, t + 1])
            for i in range(N):
                numerator = alpha[i, t] * Transition[i] * \
                            Emission[:, Observations[t + 1]].T * \
                            beta[:, t + 1].T
                xi[i, :, t] = numerator / denominator

        gamma = np.sum(xi, axis=1)
        Transition = np.sum(xi, 2) / np.sum(gamma,
                                            axis=1).reshape((-1, 1))

        # adding additional T element in gamma

        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2],
                                         axis=0).reshape((-1, 1))))

        denominator = np.sum(gamma, axis=1)
        for s in range(M):
            Emission[:, s] = np.sum(gamma[:, Observations == s],
                                    axis=1)
        Emission = np.divide(Emission, denominator.reshape((-1, 1)))
    return Transition, Emission
