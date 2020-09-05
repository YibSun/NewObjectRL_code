import torch
import modules
import torch.nn.functional as F
from torchvision.utils import save_image
from d2l import torch as d2l
import utils
import numpy as np

import torch
from torch import nn
from torch.utils import data



args = utils.args_ini()
d2l.set_figsize()

batch_size = 3
num_objects = 5
adj_full = torch.ones(num_objects, num_objects)

adj_full -= torch.eye(num_objects)
# print(adj_full)
edge_list = adj_full.nonzero()
# print(edge_list)
edge_list = edge_list.repeat(batch_size, 1)
# print(edge_list)

offset = torch.arange(0, batch_size * num_objects, num_objects).unsqueeze(-1)
offset = offset.expand(batch_size, num_objects * (num_objects - 1))
offset = offset.contiguous().view(-1)
# print(offset)
edge_list += offset.unsqueeze(-1)
edge_list = edge_list.transpose(0, 1)
# print('edge List: ')
#print(edge_list)
row, col = edge_list
# print(col)




device = torch.device('cuda' if args.cuda else 'cpu')
bach_size_default = 1024
bach_size_default = 7
dataset_default = 'data/shapes_train_s.h5'
# dataset_default = 'data/cubes_train_s.h5'
dataset = utils.StateTransitionsDataset(
    hdf5_file=dataset_default)
train_loader = data.DataLoader(
    dataset, batch_size=bach_size_default, shuffle=True, num_workers=4)

# data_batch = train_loader.__iter__().next()[0]
data_batch = train_loader.__iter__().next()
data_batch = [tensor.to(device) for tensor in data_batch]
print(data_batch[2].size())

# device = 'cpu'

learning_rate = 5e-4
input_shape = data_batch[0].size()
input_shape = torch.Size([3, 50, 50])
# print(input_shape)
model = modules.ContrastiveSWM(
    embedding_dim=args.embedding_dim,
    hidden_dim=args.hidden_dim,
    action_dim=args.action_dim,
    input_dims=input_shape,
    num_objects=args.num_objects,
    sigma=args.sigma,
    hinge=args.hinge,
    ignore_action=args.ignore_action,
    copy_action=args.copy_action,
    encoder=args.encoder).to(device)

model.apply(utils.weights_init)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.learning_rate)


class Inverse(nn.Module): 
    """MLP inverse model, maps state and next state to action.   """
    
    def __init__(self, input_dim, hidden_dim, action_dim, act_fn='relu'):
        super(Inverse, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.inverse_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            utils.get_act_fn(act_fn),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            utils.get_act_fn(act_fn),
            nn.Linear(hidden_dim, action_dim))

    def forward(self, state_pack):
        # state_pack: state and next_state: batch_size * 5 * (2+2)
        batch_s = state_pack.size(0)
        state_pack = state_pack.view(batch_s, -1)  # convert to a 1-D vector
        return self.inverse_mlp(state_pack)

class DecoderCNNSmall_test(nn.Module):
    """CNN decoder, maps latent state to image."""

    def __init__(self, input_dim, hidden_dim, num_objects, output_size,
                 act_fn='relu'):
        super(DecoderCNNSmall_test, self).__init__()

        width, height = output_size[1] // 10, output_size[2] // 10

        output_dim = width * height

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.ln = nn.LayerNorm(hidden_dim)

        self.deconv1 = nn.ConvTranspose2d(num_objects, hidden_dim,
                                          kernel_size=1, stride=1)
        self.deconv2 = nn.ConvTranspose2d(hidden_dim, output_size[0],
                                          kernel_size=10, stride=10)

        self.input_dim = input_dim
        self.num_objects = num_objects
        self.map_size = output_size[0], width, height

        self.act1 = utils.get_act_fn(act_fn)
        self.act2 = utils.get_act_fn(act_fn)
        self.act3 = utils.get_act_fn(act_fn)

    def forward(self, ins):
        h = self.act1(self.fc1(ins))
        h = self.act2(self.ln(self.fc2(h)))
        h = self.fc3(h)

        h_conv = h.view(-1, self.num_objects, self.map_size[1],
                        self.map_size[2])
        h = self.act3(self.deconv1(h_conv))
        return self.deconv2(h)


# if changing input_dim to 4, the state size should be bach_size*5*4 (two states as input)
decoder = DecoderCNNSmall_test(
    input_dim=2,
    num_objects=5,
    hidden_dim=512// 16,
    output_size=input_shape).to(device)

decoder.apply(utils.weights_init)
optimizer_dec = torch.optim.Adam(
    decoder.parameters(),
    lr=learning_rate)

obs, action, next_obs = data_batch
print('action')
print(action)

# action = action.unsqueeze(1)
# action = action.type(torch.DoubleTensor)
# action = action.to(device)
# action = torch.transpose(action, 0, -1)
print(action)
# obs_cat = torch.cat((obs, next_obs), 2)
objs = model.obj_extractor(obs)
state = model.obj_encoder(objs)
print('state size: ')
print(state)
state_conv = state.view(bach_size_default, -1)
print(state_conv)

decoder_state = decoder(state)
print('decoder state size: ')
print(decoder_state.size())

input_dim = 2
num_objects = 5
# node_attr = state.view(-1, input_dim, num_objects)
node_attr = state.view(-1)
print(node_attr)


state_cat = torch.cat((state, state), 2)
print(state_cat.size())

rec = torch.sigmoid(decoder(state))

pred_trans = model.transition_model(state, action)
# print(pred_trans.size())

num_nodes = state.size(1)
# print(action)
action_vec = utils.to_one_hot(
    action, args.action_dim * num_nodes)
# print(action_vec)
action_vec = action_vec.view(-1, args.action_dim)
# print(action_vec)


inv_model = Inverse(
    input_dim=20,
    hidden_dim=256,
    action_dim=1).to(device)

inv_model.apply(utils.weights_init)
optimizer_inv = torch.optim.Adam(
    inv_model.parameters(),
    lr=learning_rate)
# state_cat = state_cat[1]
print('state cat')
# print(state_cat)
pred_actions = inv_model(state_cat)
print(pred_actions)
pred_ac_copy = pred_actions.clone()
fill_num = pred_ac_copy[3].item()
print('fill num ')
print(fill_num)


def fill_act(actions, index):
    actions_copy = actions.clone()
    fill_number = actions_copy[index].item()
    actions_copy = actions_copy.fill_(fill_number)
    return actions_copy

pred_ac_copy = fill_act(pred_ac_copy, 3)
# action_cat = torch.cat((pred_actions[0], pred_actions[0]), -1)
print(pred_ac_copy)
loss = []
for i in range(pred_actions.size(0)):
    act = fill_act(pred_actions, i)
    if i == 1:
        loss1 = F.mse_loss(pred_actions, act) / pred_actions.size(0)
        loss.append(loss1)

    if i == 2:
        loss2 = F.mse_loss(pred_actions, act) / pred_actions.size(0)
        loss.append(loss2)
loss_sum = sum(loss)

# Train model.
print('Starting model training...')
step = 0
best_loss = 1e9

for epoch in range(1, args.epochs + 1):
    inv_model.train()
    train_loss = 0
    for batch_idx, data_batch in enumerate(train_loader):
        data_batch = [tensor.to(device) for tensor in data_batch]
        optimizer_inv.zero_grad()
        obs, action, next_obs = data_batch
        objs = model.obj_extractor(obs)
        next_objs = model.obj_extractor(next_obs)
        state = model.obj_encoder(objs)
        next_state = model.obj_encoder(next_objs)

        state_cat = torch.cat((state, next_state), 2)

        pred_actions = inv_model(state_cat)
        loss_cont = []
        for i in range(pred_actions.size(0)):
            act = fill_act(pred_actions, i)
            loss_cur = F.mse_loss(pred_actions, act) / pred_actions.size(0)
            loss_cont.append(loss_cur)
        ave_loss = sum(loss_cont) / pred_actions.size(0)

        zeros = torch.zeros_like(ave_loss)
        # ave_loss = torch.max(zeros, 1 - ave_loss)
        # ave_loss = torch.max(zeros, -ave_loss)
        ave_loss = -ave_loss

        action = action.unsqueeze(1)
        action = action.type(torch.cuda.FloatTensor)
        loss = F.mse_loss(pred_actions, action) / obs.size(0)
        loss = loss + ave_loss
        loss.backward()  # compute new gradients
        optimizer_inv.step()  # perform one gradient descent step

        train_loss += loss.item()

    avg_loss = train_loss / len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.6f}'.format(
        epoch, avg_loss))
    # print(state.size())




''' 
for epoch in range(1, args.epochs + 1):
    decoder.train()
    train_loss = 0

    for batch_idx, data_batch in enumerate(train_loader):
        data_batch = [tensor.to(device) for tensor in data_batch]
        optimizer.zero_grad()

        optimizer_dec.zero_grad()
        obs, action, next_obs = data_batch

        objs = model.obj_extractor(obs)
        state = model.obj_encoder(objs)
        rec = torch.sigmoid(decoder(state))
        loss = F.binary_cross_entropy(
            rec, obs, reduction='sum') / obs.size(0)
        next_state_pred = state + model.transition_model(state, action)
        next_rec = torch.sigmoid(decoder(next_state_pred))
        next_loss = F.binary_cross_entropy(
            next_rec, next_obs,
            reduction='sum') / obs.size(0)

        loss += next_loss

        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        optimizer_dec.step()
    avg_loss = train_loss / len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.6f}'.format(
        epoch, avg_loss))
    print(state.size())
    # save_image(obs[0], 'img2.jpg')
'''


print('done')