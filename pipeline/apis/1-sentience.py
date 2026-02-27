#!/usr/bin/env python3
"""
Module that queries the Star Wars API (SWAPI) to find
the home planets of all sentient species.
"""

import requests


def sentientPlanets():
    """
    Returns a list of home planet names of all sentient species.

    Sentient type is determined by checking if the string
    "sentient" appears in either the classification or
    designation attributes.

    Returns:
        list: Names of home planets.
    """
    planets = []
    url = "https://swapi.dev/api/species/"

    while url:
        response = requests.get(url)
        if response.status_code != 200:
            break

        data = response.json()

        for species in data.get("results", []):
            classification = species.get("classification", "").lower()
            designation = species.get("designation", "").lower()

            if "sentient" in classification or "sentient" in designation:
                homeworld_url = species.get("homeworld")

                if homeworld_url:
                    planet_response = requests.get(homeworld_url)
                    if planet_response.status_code == 200:
                        planet_data = planet_response.json()
                        planets.append(planet_data.get("name"))

        url = data.get("next")

    return planets
