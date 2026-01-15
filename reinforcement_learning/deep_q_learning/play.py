#!/usr/bin/env python3
"""
Play a trained Deep Q-Network agent on Atari Breakout.
"""

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory


def build_env():
    """
    Build the Atari Breakout environment with the SAME wrappers
    used during training.
    """
    env = gym.make("ALE/Breakout-v5", render_mode="human")
    env = AtariPreprocessing(
        env,
        grayscale_obs=True,
        scale_obs=True,
        frame_skip=4
    )
    env = FrameStack(env, 4)
    return env


def build_model(actions):
    """
    Build the SAME Q-network architecture used during training.
    """
    model = Sequential()
    model.add(
        Conv2D(
            32,
            (8, 8),
            strides=(4, 4),
            activation="relu",
            input_shape=(84, 84, 4),
        )
    )
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation="relu"))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(actions, activation="linear"))
    return model


def main():
    """
    Load trained weights and play one episode.
    """
    env = build_env()
    actions = env.action_space.n

    model = build_model(actions)

    memory = SequentialMemory(limit=1, window_length=4)
    policy = GreedyQPolicy()

    dqn = DQNAgent(
        model=model,
        nb_actions=actions,
        memory=memory,
        policy=policy,
        nb_steps_warmup=0,
        gamma=0.99,
        target_model_update=10_000,
    )

    dqn.compile(Adam(learning_rate=1e-4), metrics=["mae"])

    # ✅ THIS WILL NOW LOAD CORRECTLY
    dqn.load_weights("policy.h5")

    # ❗ visualize=False is REQUIRED (Gymnasium compatibility)
    dqn.test(env, nb_episodes=1, visualize=False)

    env.close()


if __name__ == "__main__":
    main()
