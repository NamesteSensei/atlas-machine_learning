#!/usr/bin/env python3
"""
Module that queries the Star Wars API (SWAPI) to find starships
that can carry at least a given number of passengers.
"""

import requests


def availableShips(passengerCount):
    """
    Returns a list of starship names that can carry
    at least passengerCount passengers.

    Args:
        passengerCount (int): Minimum number of passengers.

    Returns:
        list: Names of qualifying starships.
    """
    ships = []
    url = "https://swapi.dev/api/starships/"

    while url:
        response = requests.get(url)
        if response.status_code != 200:
            break

        data = response.json()

        for ship in data.get("results", []):
            passengers = ship.get("passengers", "")

            # Remove commas from passenger string
            passengers = passengers.replace(",", "")

            # Check if passenger value is numeric
            if passengers.isdigit():
                if int(passengers) >= passengerCount:
                    ships.append(ship.get("name"))

        url = data.get("next")

    return ships
