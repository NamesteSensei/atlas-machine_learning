#!/usr/bin/env python3
"""
Script that displays the first SpaceX launch with:
- Launch name
- Date (local time)
- Rocket name
- Launchpad name and locality
"""

import requests


def get_first_launch():
    """
    Fetches and prints the first SpaceX launch details.
    """
    launches_url = "https://api.spacexdata.com/v4/launches"
    rockets_url = "https://api.spacexdata.com/v4/rockets/"
    launchpads_url = "https://api.spacexdata.com/v4/launchpads/"

    response = requests.get(launches_url)
    if response.status_code != 200:
        return

    launches = response.json()

    # Sort by date_unix ascending
    launches_sorted = sorted(launches, key=lambda x: x.get("date_unix", 0))

    first_launch = launches_sorted[0]

    launch_name = first_launch.get("name")
    launch_date = first_launch.get("date_local")
    rocket_id = first_launch.get("rocket")
    launchpad_id = first_launch.get("launchpad")

    # Fetch rocket name
    rocket_response = requests.get(rockets_url + rocket_id)
    rocket_name = ""
    if rocket_response.status_code == 200:
        rocket_name = rocket_response.json().get("name")

    # Fetch launchpad details
    launchpad_response = requests.get(launchpads_url + launchpad_id)
    launchpad_name = ""
    launchpad_locality = ""
    if launchpad_response.status_code == 200:
        launchpad_data = launchpad_response.json()
        launchpad_name = launchpad_data.get("name")
        launchpad_locality = launchpad_data.get("locality")

    print(
        f"{launch_name} ({launch_date}) "
        f"{rocket_name} - {launchpad_name} ({launchpad_locality})"
    )


if __name__ == '__main__':
    get_first_launch()
