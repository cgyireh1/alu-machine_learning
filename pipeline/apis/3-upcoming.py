#!/usr/bin/env python3
"""
Script that displays upcoming launch with these information:
Name of the launch
The date (in local time)
The rocket name
The name (with the locality) of the launchpad 
"""

import requests

def get_upcoming_launch():
    url = 'https://api.spacexdata.com/v4/launches/upcoming'
    response = requests.get(url)
    response.raise_for_status()

    upcoming_launch = min(response.json(), key=lambda x: x['date_unix'])

    return upcoming_launch

def get_rocket_name(rocket_id):
    rocket_url = f"https://api.spacexdata.com/v4/rockets/{rocket_id}"
    response = requests.get(rocket_url)
    response.raise_for_status()
    return response.json()['name']

def get_launchpad_info(launchpad_id):
    launchpad_url = f"https://api.spacexdata.com/v4/launchpads/{launchpad_id}"
    response = requests.get(launchpad_url)
    response.raise_for_status()
    return response.json()

if __name__ == '__main__':
    launch = get_upcoming_launch()
    launch_name = launch['name']
    date_local = launch['date_local']
    rocket_name = get_rocket_name(launch['rocket'])
    launchpad_info = get_launchpad_info(launch['launchpad'])
    launchpad_name = launchpad_info['name']
    launchpad_locality = launchpad_info['locality']

    upcoming_launch_info = f"{launch_name} ({date_local}) {rocket_name} - {launchpad_name} ({launchpad_locality})"
    
    print(upcoming_launch_info)
