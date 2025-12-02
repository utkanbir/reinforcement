import gymnasium as gym
from gym_super_mario_bros import make as mario_make
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from gymnasium.wrappers import FrameStack
import cv2
import numpy as np
import time

# Aynı wrapper
class GrayScaleResize(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

    def observation(self, obs):
        frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        return np.expand_dims(frame, -1)

# Env
env = mario_make("SuperMarioBros-1-1-v3")
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GrayScaleResize(env)
env = FrameStack(env, 4)

model = PPO.load("mario_ppo")

obs = env.reset()
done = False

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)

    frame = env.render()

    cv2.imshow("Mario PPO Agent", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

env.close()
cv2.destroyAllWindows()
