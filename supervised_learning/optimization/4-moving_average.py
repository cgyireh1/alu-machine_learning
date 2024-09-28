#!/usr/bin/env python3
""" Moving Average """


def moving_average(data, beta):
    """
    data is the list of data to calculate the moving average of
    beta is the weight used for the moving average
    Your moving average calculation should use bias correction
    """

    Var = 0
    Moving_avg = []
    for i in range(len(data)):
        Var = (beta * Var) + ((1 - beta) * data[i])
        bias_correction_avg = Var / 1 - beta ** (i + 1)
        Moving_avg.append(bias_correction_avg)
    return Moving_avg
