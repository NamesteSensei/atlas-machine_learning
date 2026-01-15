#!/usr/bin/env python3
"""
Train a Deep Q-Network agent to play Atari Breakout using keras-rl2.
"""

import gymnasium as gym

from gymnasium.wrappers import AtariPreprocessing

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Permute
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory


class GymnasiumToGymWrapper:
    """
    Adapter to make Gymnasium environments compatible with keras-rl2.
    """

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self):
        obs, _ = self.env.reset()
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


def build_env():
    """
    Build Atari Breakout environment.
    No FrameStack — keras-rl2 handles stacking.
    """
    env = gym.make(
        "ALE/Breakout-v5",
        frameskip=1,
        render_mode=None
    )

    env = AtariPreprocessing(
        env,
        grayscale_obs=True,
        scale_obs=True,
        frame_skip=4
    )

    return GymnasiumToGymWrapper(env)


def build_model(nb_actions):
    """
    Deep Q-Network architecture from the DQN paper.
    """

    model = Sequential()

    # 🔑 CRITICAL FIX: convert (4, 84, 84) -> (84, 84, 4)
    model.add(Permute((2, 3, 1), input_shape=(4, 84, 84)))

    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation="relu"))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation="relu"))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(nb_actions, activation="linear"))

    return model


def main():
    env = build_env()

    nb_actions = env.action_space.n

    model = build_model(nb_actions)

    memory = SequentialMemory(limit=200_000, window_length=4)
    policy = EpsGreedyQPolicy()

    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        policy=policy,
        nb_steps_warmup=10_000,
        target_model_update=10_000
    )

    dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])

    dqn.fit(env, nb_steps=25_000, visualize=False, verbose=2)

    dqn.save_weights("policy.h5", overwrite=True)

    env.close()


if __name__ == "__main__":
    main()
