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
            rocketDict[rocket] += 1
    rocketList = sorted(rocketDict.items(), key=lambda kv: kv[0])
    rocketList = sorted(rocketList, key=lambda kv: kv[1], reverse=True)
    for rocket in rocketList:
        print("{}: {}".format(rocket[0], rocket[1]))
