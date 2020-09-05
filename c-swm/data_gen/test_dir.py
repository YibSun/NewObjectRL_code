# Get env directory
import sys
from pathlib import Path
if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

from utils import save_list_dict_h5py
import envs
from envs import physics_sim
import gym
import utils
from torch.utils import data

# env_id = 'ShapesTrain-v0'
env_id = 'SpaceInvaders-v0'

env = gym.make(env_id)
env.reset()
for _ in range(10000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()

print("done")
