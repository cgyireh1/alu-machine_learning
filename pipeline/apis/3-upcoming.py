#!/usr/bin/env python3
"""
Script that displays upcoming launch with these information:

Name of the launch
The date (in local time)
The rocket name
The name (with the locality) of the launchpad 
"""

import requests

if __name__ == '__main__':
    rl = "https://api.spacexdata.com/v4/launches/upcoming"
    results = requests.get(url).json()
    dateCheck = float('inf')
    launchName = None
    rocket = None
    launchPad = None
    location = None
    for launch in results:
        launchDate = launch.get('date_unix')
        if launchDate < dateCheck:
            dateCheck = launchDate
            date = launch.get('date_local')
            launchName = launch.get('name')
            rocket = launch.get('rocket')
            launchPad = launch.get('launchpad')
    if rocket:
        rocket = requests.get('https://api.spacexdata.com/v4/rockets/{}'.
                              format(rocket)).json().get('name')
    if launchPad:
        launchpad = requests.get('https://api.spacexdata.com/v4/launchpads/{}'.
                                 format(launchPad)).json()
        launchPad = launchpad.get('name')
        location = launchpad.get('locality')

    print("{} ({}) {} - {} ({})".format(
        launchName, date, rocket, launchPad, location))
    lpad_id = launch['launchpad']
    lpad_url = "https://api.spacexdata.com/v4/launchpads/{}".\
        format(lpad_id)
    lpad_req = requests.get(lpad_url).json()
    lpad_name = lpad_req['name']
    lpad_loc = lpad_req['locality']
    
    upcoming_launch = "{} ({}) {} - {} ({})".format(launch_name, date_l,
                                                    rocket_name, lpad_name,
                                                    lpad_loc)
    
    print(upcoming_launch)
