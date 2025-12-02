import gym
from gym import Wrapper
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros import make as mario_make
from nes_py.wrappers import JoypadSpace
import numpy as np
import cv2
import time

# Compatibility wrapper to convert old gym API (4 values) to new API (5 values)
class OldToNewAPIWrapper(Wrapper):
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if isinstance(obs, tuple):
            return obs
        return obs, {}
    
    def step(self, action):
        result = self.env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            # Convert old API to new API: (obs, reward, done, info) -> (obs, reward, terminated, truncated, info)
            return obs, reward, done, False, info
        return result

# Mario environment
env = mario_make("SuperMarioBros-1-1-v0")

# Patch time_limit wrapper to handle old API (4 values instead of 5)
def patch_time_limit_wrapper(env):
    """Recursively find and patch TimeLimit wrappers"""
    if hasattr(env, 'env'):
        patch_time_limit_wrapper(env.env)
    
    # Check if this is a TimeLimit wrapper
    if 'TimeLimit' in str(type(env)):
        # Store original step method
        original_inner_step = env.env.step
        
        def patched_step(action):
            # Call inner environment directly with old API
            inner_result = original_inner_step(action)
            if len(inner_result) == 4:
                obs, reward, done, info = inner_result
                env._elapsed_steps += 1
                truncated = env._elapsed_steps >= env._max_episode_steps
                terminated = done and not truncated
                return obs, reward, terminated, truncated, info
            return inner_result
        
        env.step = patched_step

# Apply patch
patch_time_limit_wrapper(env)

# Wrap with compatibility layer
env = OldToNewAPIWrapper(env)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

obs, info = env.reset()

# Sürekli oyunu göster
done = False

while not done:
    # Random action (henüz RL yok)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # Use observation directly as frame (render() may not work with old gym API)
    frame = obs.copy() if isinstance(obs, np.ndarray) else obs
    
    # Ensure frame is a valid numpy array
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        continue
    
    # Convert RGB to BGR for OpenCV (OpenCV uses BGR format)
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Resize if needed for better display (optional)
    if frame.shape[0] > 0 and frame.shape[1] > 0:
        # Scale up for better visibility (Mario is 240x256, scale 2x)
        frame = cv2.resize(frame, (frame.shape[1] * 2, frame.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Super Mario RL", frame)

    # ESC basılırsa çık
    if cv2.waitKey(15) & 0xFF == 27:
        break

env.close()
cv2.destroyAllWindows()
