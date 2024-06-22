#!/usr/bin/env python3
"""
Function that calculates the posterior probability that the
various hypothetical probabilities of developing severe side effects
falls within a specific range given the data
"""


from scipy import special


def posterior(x, n, p1, p2):
    """
    x [int]: total number of patients that develop severe side effects
    n [int]: total number of patients observed
    p1 [float]: the lower bound on the range
    p2 [float]: the upper bound on the range
    Assume prior beliefs of p follow a uniform distribution
    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(p1) is not float or p1 < 0 or p1 > 1:
        raise ValueError("p1 must be a float in the range [0, 1]")
    if type(p2) is not float or p2 < 0 or p2 > 1:
        raise ValueError("p2 must be a float in the range [0, 1]")
    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")
    # difference between cumulative distribution
    #function for beta distribution from 0 to p2 and p1
    posterior = special.betainc(x + 1, n - x + 1, p2)
    - special.betainc(x + 1, n - x + 1, p1)
    return posterior
