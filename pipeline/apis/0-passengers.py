#!/usr/bin/env python3
""" Number of passengers"""
import requests


def availableShips(passengerCount):
    """
    By using the Swapi API, create a method that
    returns the list of ships that can hold a
    given number of passengers:

    Prototype: def availableShips(passengerCount):
    Don’t forget the pagination
    If no ship available, return an empty list.
    """
    starships = []
    url = 'https://swapi-api.alx-tools.com/api/starships/'
    while url is not None:
        response = requests.get(url,
                                headers={'Accept': 'application/json'},
                                params={"term": 'starships'})
        for ship in response.json()['results']:
            passenger = ship['passengers'].replace(',', '')
            if passenger.isnumeric() and int(passenger) >= passengerCount:
                starships.append(ship['name'])
        url = response.json()['next']
    return starships
