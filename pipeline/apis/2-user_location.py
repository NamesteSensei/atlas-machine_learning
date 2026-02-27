#!/usr/bin/env python3
"""
Script that prints the location of a specific GitHub user
using the GitHub API.
"""

import requests
import sys
import time


def get_user_location(url):
    """
    Retrieves and prints the location of a GitHub user.

    Args:
        url (str): Full GitHub API URL for the user.
    """
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        print(data.get("location"))

    elif response.status_code == 404:
        print("Not found")

    elif response.status_code == 403:
        reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
        current_time = int(time.time())
        minutes = (reset_time - current_time) // 60
        print(f"Reset in {minutes} min")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(1)

    get_user_location(sys.argv[1])
