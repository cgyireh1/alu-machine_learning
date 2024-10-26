#!/usr/bin/env python3
"""
Script that displays the number of launches per rocket
using the (unofficial) SpaceX API
"""
import requests


if __name__ == '__main__':
    url_l = 'https://api.spacexdata.com/v4/launches'
    results_l = requests.get(url_l).json()
    rocketDict = {}
    for launch in results_l:
        rocket = launch.get('rocket')
        url_r = 'https://api.spacexdata.com/v4/rockets/{}'.format(rocket)
        results_l = requests.get(url_r).json()
        rocket = results_l.get('name')
        if rocketDict.get(rocket) is None:
            rocketDict[rocket] = 1
        else:
            object[rocket] = 1

    keys = sorted(object.items(), key=lambda x: x[0])
    keys = sorted(keys, key=lambda x: x[1], reverse=True)

    for k in keys:
        print("{}: {}".format(k[0], k[1]))
