# Code to run experiments on Oddssey
import sys
############# ARGUMENT STRUCTURE ##################
_, tar_cost, gamma, al_itr, n_exp, rl_exps, use_cnn = sys.argv
# Type conversions 
tar_cost = float(tar_cost)
gamma = float(gamma)
al_itr = int(al_itr)
n_exp = int(n_exp)
rl_exps = int(rl_exps)
use_cnn = use_cnn=='True'
print('Using Values:')
print(tar_cost, gamma, al_itr, n_exp, rl_exps, use_cnn)
##################################################
# Imports 
# Code to run experiments on Oddssey
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns
from tqdm import tqdm_notebook as tqdm
import torchvision.models as tmodels
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.distributions import Categorical
# My custom files
import active_learning as al
import reinforcement as rl
import data as d 

# Get raw datasets - MNIST
train_set = dset.MNIST(root='./data', train=True, transform=transforms.ToTensor(),download=True)
test_set = dset.MNIST(root='./data', train=False, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=len(test_set),shuffle=False)

# Get raw dataset - USPS
percent_test = 0.3
usps_batch = 64
usps_set = d.get_usps('usps_all.mat', size=(28,28))
usps_x, usps_y, usps_test_x, usps_test_y = al.get_dataset_split(usps_set,int(len(usps_set)*percent_test))
usps_test_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(usps_test_x, usps_test_y), \
                                               batch_size=len(usps_test_y),shuffle=False)
usps_train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(usps_x, usps_y), \
                                               batch_size=usps_batch,shuffle=True)

train_x, train_y, val_x, val_y = al.get_dataset_split(train_set)
test_x,test_y = al.get_xy_split(test_loader)

# Make the RL agent to interact with the environment
class AgentRL(nn.Module):
    def __init__(self, inpt_dim, hidden_dim, num_policies):
        super(AgentRL, self).__init__()
        self.num_policies = num_policies
        self.inner_layer = nn.Linear(inpt_dim, hidden_dim)
        self.outer_layer = nn.Linear(hidden_dim, num_policies)
        self.rewards = []
        self.saved_log_probs = []

    def forward(self, x):
        x = x.view(1,-1)
        x = F.relu(self.inner_layer(x))
        x = self.outer_layer(x)
        return F.softmax(x, dim=1)

# Define the logistic regression model
class logreg(nn.Module):
    """ Logistic regression """
    def __init__(self):
        super(logreg, self).__init__()
        self.classes = 10
        self.w = nn.Linear(28*28,self.classes)

    def forward(self, x):
        x = self.w(x.view(-1,1,28*28))
        return F.log_softmax(x.view(-1,self.classes),dim=1)

# Define the CNN model
class CNN(nn.Module):
    """ CNN for MNIST """
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 10)

    def forward(self, x):
        x = x.view(-1,1,28,28)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = self.fc1(x)
        return F.log_softmax(x,dim=1)


tar_costs = [tar_cost]*rl_exps
pc_all, rwd_all, ac_all = [],[],[]

for tar_c in tar_costs:
    if use_cnn:
        mod = CNN()
    else:
        mod = logreg()
    opt = optim.Adam(mod.parameters(),lr=0.01)
    policy_key = {0: 'transfer', 1: 'boundary'}
    agent = AgentRL(int(len(train_x)*10),128, 2) # 2 for the 1 AL policy and one TL policy
    optimizer_rl = optim.Adam(agent.parameters(), lr=1e-2)

    rl_e = rl.Environment(mod, train_x, train_y, val_x , val_y,  nn.NLLLoss(), opt, usps_data=usps_train_loader)
    rl_e.set_params(al_itrs=al_itr, npoints=20, batch_size=10)
    pc,rwd,ac = rl_e.run_experiments(agent, optimizer_rl, policy_key, n_experiments=n_exp,  \
                                     gamma=gamma,tar_cost=tar_c,rtype='transfer')
    pc_all.append(pc)
    rwd_all.append(rwd)
    ac_all.append(ac)


with open(f"rl_results_lr_len{al_itr}_tc{tar_cost}_gamma{gamma}.pkl", "wb" ) as file:
    pickle.dump({'policy': pc_all,'reward': rwd_all, 'acc': ac_all},file,protocol=pickle.HIGHEST_PROTOCOL)
