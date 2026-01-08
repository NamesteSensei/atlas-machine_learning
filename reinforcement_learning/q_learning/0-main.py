#!/usr/bin/env python3
"""Test cases for the load_frozen_lake function."""

load_frozen_lake = __import__('0-load_env').load_frozen_lake

# Test default 8x8 map, not slippery
env = load_frozen_lake()
print(env.unwrapped.desc)
print(len(env.unwrapped.P[0][0]))
print(env.unwrapped.P[0][0])

# Test default 8x8 map, slippery
env = load_frozen_lake(is_slippery=True)
print(env.unwrapped.desc)
print(len(env.unwrapped.P[0][0]))
print(env.unwrapped.P[0][0])

# Test custom 3x3 map
desc = [['S', 'F', 'F'],
        ['F', 'H', 'H'],
        ['F', 'F', 'G']]
env = load_frozen_lake(desc=desc)
print(env.unwrapped.desc)

# Test pre-made 4x4 map
env = load_frozen_lake(map_name='4x4')
print(env.unwrapped.desc)
