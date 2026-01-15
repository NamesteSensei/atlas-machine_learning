#!/usr/bin/env python3
"""Main file to train REINFORCE and save score plot."""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import random
train = __import__('train').train


def set_seed(env, seed=0):
    """Set random seeds for reproducibility."""
    env.reset(seed=seed)
    np.random.seed(seed)
    random.seed(seed)


# Setup environment
env = gym.make('CartPole-v1')
set_seed(env, 0)

# Train the agent
scores = train(env, 10000)

# Plot and save the scores to a file
plt.plot(np.arange(len(scores)), scores)
plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("REINFORCE Training Performance")
plt.grid(True)
plt.savefig("scores.png")  # Save to file instead of showing
print("Saved plot to scores.png")

env.close()
