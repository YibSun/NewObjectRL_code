import numpy as np
import torch
import envs

USE_CUDA = torch.cuda.is_available()

from torchvision.utils import save_image
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
from torchvision.transforms import ToPILImage
from IPython.display import Image
import matplotlib.pyplot as plt
import torchvision

x = torch.randn(10, 2)
print(x)
zero_con = torch.zeros(10, 5)
print(zero_con)
node_attr = torch.cat([x, zero_con], dim=1)
print(node_attr)

# torch.cat([source, target], dim=1)


# env_id = 'ShapesTrain-v0'
env_id = 'SpaceInvaders-v0'


env = gym.make(env_id)
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()

bach_size_default = 1024
dataset_default = 'data/shapes_train_s.h5'
dataset = utils.StateTransitionsDataset(
    hdf5_file=dataset_default)
train_loader = data.DataLoader(
    dataset, batch_size=bach_size_default, shuffle=True, num_workers=4)

obs = train_loader.__iter__().next()[0]
device = torch.device('cuda')
device = torch.device('cpu')
for batch_idx, data_batch in enumerate(train_loader):
    data_batch = [tensor.to(device) for tensor in data_batch]

print('data_batch obs size: ')
print(data_batch[0].size())
print('data_batch action size: ')
print(data_batch[1].size())

print(data_batch[1][0])

action = data_batch[1][0]
action = data_batch[1]
action_vec = utils.to_one_hot(action, 4 * 5)
action_vec = action_vec.view(-1, 4)
print(action_vec[0])
print(action_vec[1])
print(action_vec[2])
print(action_vec[3])
print(action_vec[4])



to_img = ToPILImage()
# img = to_img(obs[0])
img = to_img(data_batch[0][0])
img2 = to_img(data_batch[2][0])

plt.imshow(img)
plt.show()
plt.imshow(img2)
plt.show()

# save_image(data_batch[0][6], 'img2.jpg')

input_shape = obs[0].size()
input_shape = obs.size()

print('input shape', input_shape)

print("done")