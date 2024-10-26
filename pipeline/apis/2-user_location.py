#!/usr/bin/env python3
""" Get requests location from Github API"""

import sys
import requests
import time


if __name__ == '__main__':
    """
    write a script that prints the location of a specific
    user using the GitHub API
    """
    url = sys.argv[1]
    response = requests.get(url)
    if response.status_code == 200:
        print(response.json()['location'])
    elif response.status_code == 404:
        print('Not found')
    elif response.status_code == 403:
        limit = int(response.headers.get('X-Ratelimit-Reset'))
        start_time = int(time.time())
        elapsed_time = int((limit - start_time) / 60)
        print('Reset in {} min'.format(int(elapsed_time)))
