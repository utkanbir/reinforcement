import gymnasium as gym
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros import make as mario_make
from nes_py.wrappers import JoypadSpace
import numpy as np
import cv2
import time

# Mario environment
env = mario_make("SuperMarioBros-1-1-v0")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

obs, info = env.reset()

# Sürekli oyunu göster
done = False

while not done:
    # Random action (henüz RL yok)
    action = env.action_space.sample()
    obs, reward, done, trunc, info = env.step(action)

    frame = env.render()  # Burası gerçek zamanlı pencere açar

    cv2.imshow("Super Mario RL", frame)

    # ESC basılırsa çık
    if cv2.waitKey(15) & 0xFF == 27:
        break

env.close()
cv2.destroyAllWindows()
