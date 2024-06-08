#!/usr/bin/env python3
""" Poisson class that represents Poisson distribution """


class Poisson:
    """
    initialize poisson
    calculates pmf
    calculates cdf
    """

    def __init__(self, data=None, lambtha=1.):
        """
        class constructor:
        def __init__(self, data=None, lambtha=1.):
        - data is a list of the data to be used to estimate the distribution
        - lambtha is the expected number of occurrences in a given time frame
        - Sets the instance attribute lambtha
        - Saves lambtha as a float
        - If data is not given, (i.e. None) data is an empty list
        - If data is given, calculates the lambtha of data
        - Raises a TypeError if data is not a list
        - Raises ValueError If data does not contain at least two data points
        """
        if data is None:
            if lambtha < 1:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                lambtha = float(sum(data) / len(data))
                self.lambtha = lambtha

    def pmf(self, k):
        """
        Update the poisson class
        calculates the value of the PMF for a given number of successes
            k [int]: number of successes
                If k is not an int, convert it to int
                If k is out of range, return 0
        """
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        e = 2.7182818285
        lambtha = self.lambtha
        factorial = 1
        for i in range(k):
            factorial *= (i + 1)
        pmf = ((lambtha ** k) * (e ** -lambtha)) / factorial
        return pmf

    def cdf(self, k):
        """
        Update the poisson class
        calculates the value of the CDF for a given number of successes
        k [int]: number of successes
          If k is not an int, convert it to int
          If k is out of range, return 0
        """
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf 