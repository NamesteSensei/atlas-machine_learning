#!/usr/bin/env python3
"""Module to load the FrozenLake environment using gymnasium."""

import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Load the FrozenLake environment.

    Args:
        desc (list[list[str]] or None): Custom map description.
        map_name (str or None): Name of a pre-made map (e.g., '4x4').
        is_slippery (bool): If True, the environment has slippery ice.

    Returns:
        gym.Env: The initialized FrozenLake environment.
    """
    env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery
    )
    return env
