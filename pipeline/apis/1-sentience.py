#!/usr/bin/env python3
""" Sentient species"""
import requests


def sentientPlanets():
    """
    By using the Swapi API, create a method that 
    returns the list of names of the home 
    planets of all sentient species.
    Prototype: def sentientPlanets():
    Don’t forget the pagination
    """
    planets = []
    url = 'https://swapi-api.alx-tools.com/api/species/'
    while url is not None:
        response = requests.get(url,
                                headers={'Accept': 'application/json'},
                                params={"term": 'specie'})
        for specie in response.json()['results']:
            if specie['classification'] == 'sentient' or \
                    specie['designation'] == 'sentient':
                if specie['homeworld'] is not None:
                    homeworld = requests.get(specie['homeworld'])
                    planets.append(homeworld.json()['name'])

        url = response.json()['next']
    return planets