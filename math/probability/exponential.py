#!/usr/bin/env python3
""" Exponential class that represents exponential distribution """


class Exponential:
    """
    initialize exponential class
    calculates pmf for a given time period
    calculates cdf for a given time period
    """

    def __init__(self, data=None, lambtha=1.):
        """
        class that represents exponential distribution
        class constructor:
        def __init__(self, data=None, lambtha=1.)
          - data is a list of the data to be used to estimate the distribution
          - lambtha is the expected number of occurrences in a given time frame
          - Sets the instance attribute lambtha
            - saves lambtha as a float
          - if data is not given:  
            - use the given lambtha 
            - raise ValueError if lambtha is not positive value
          - if data is given:
            - calculate the lambtha of data
            - raise TypeError if data is not a list
            - raise ValueError if data does not contain at least two data points  
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
                lambtha = float(len(data) / sum(data))
                self.lambtha = lambtha

    def pdf(self, x):
        """
        calculates the value of the PDF for a given time period
          x is the time period
          Return the PDF value for x
          If x is out of range, return 0
        """
        if x < 0:
            return 0
        e = 2.7182818285
        lambtha = self.lambtha
        pdf = lambtha * (e ** (-lambtha * x))
        return pdf

    def cdf(self, x):
        """
        calculates the value of CDF for a given time period
          x is the time period
          Return the CDF value for x
          If x is out of range, return 0
        """
        if x < 0:
            return 0
        e = 2.7182818285
        lambtha = self.lambtha
        cdf = 1 - (e ** (-lambtha * x))
        return cdf