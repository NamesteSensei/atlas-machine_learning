#!/usr/bin/env python3
"""
Script that displays the number of SpaceX launches per rocket.
"""

import requests


def get_rocket_frequencies():
    """
    Fetches all launches and prints the number of launches per rocket.
    Results are sorted by:
    - Number of launches (descending)
    - Rocket name (alphabetical order if tied)
    """
    launches_url = "https://api.spacexdata.com/v4/launches"
    rockets_url = "https://api.spacexdata.com/v4/rockets/"

    response = requests.get(launches_url)
    if response.status_code != 200:
        return

    launches = response.json()

    rocket_counts = {}

    # Count launches per rocket ID
    for launch in launches:
        rocket_id = launch.get("rocket")
        if rocket_id:
            rocket_counts[rocket_id] = (
                rocket_counts.get(rocket_id, 0) + 1
            )

    rocket_data = []

    # Convert rocket IDs to rocket names
    for rocket_id, count in rocket_counts.items():
        rocket_response = requests.get(rockets_url + rocket_id)
        if rocket_response.status_code == 200:
            rocket_name = rocket_response.json().get("name")
            rocket_data.append((rocket_name, count))

    # Sort by count descending, then name ascending
    rocket_data_sorted = sorted(
        rocket_data,
        key=lambda x: (-x[1], x[0])
    )

    for name, count in rocket_data_sorted:
        print(f"{name}: {count}")


if __name__ == '__main__':
    get_rocket_frequencies()
