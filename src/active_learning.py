# Imports
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
from tqdm import tqdm_notebook as tqdm
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
# from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
# import pyro


def _get_samplers(total_len,init_size,val_size=10000, random_seed=1492):
    """ Get docs at some point, but this basically just get the samplers for initialization"""
    if random_seed:
        np.random.seed(random_seed)
    # Get Indices for validation and for initial random training sample
    idxs = list(range(len(train_set)))
    val_idx = np.random.choice(idxs,size=val_size, replace=False)
    train_idx = list(set(idxs)-set(val_idx))
    init_labels_idx = np.random.choice(train_idx,size=init_size, replace=False)
    unlabeled_idx = list(set(idxs)-set(init_labels_idx))

    # Get samplers for torch
    val_sampler = SubsetRandomSampler(val_idx)
    train_sampler = SubsetRandomSampler(init_labels_idx)
    unlabeled_sampler =  SubsetRandomSampler(unlabeled_idx)
    return train_sampler, unlabeled_sampler, val_sampler, init_labels_idx, unlabeled_idx

def setup_data_loaders(batch_size=8, starting_size=64, val_size=10000, use_cuda=False):
    """ Function to get the dataloaders for the train, unlabeled, validation and
    test datasets.  It takes in the parameters for the starting sizes and batch sizes,
    along with arguments for the validation set size and the whether to use CUDA.

    NOTE: This is ONLY for the MNIST dataset.
    """
    root = './data'
    download = True
    trans = transforms.ToTensor()
    train_set = dset.MNIST(root=root, train=True, transform=trans,
                           download=download)
    test_set = dset.MNIST(root=root, train=False, transform=trans)

    tr_samp, unlab_samp, val_samp, init_idx, unlab_idx = _get_samplers(len(train_set), starting_size, val_size)

    kwargs = {'num_workers': 1, 'pin_memory': use_cuda}
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, \
                                               sampler=tr_samp, **kwargs)
    unlab_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=len(unlab_idx), \
                                               sampler=unlab_samp, **kwargs)
    val_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=val_size, \
                                             sampler=val_samp, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=len(test_set), \
                                              shuffle=False, **kwargs)
    return (train_loader, unlab_loader, val_loader, test_loader), init_idx, unlab_idx

# Get the x,y seperation
def get_xy_split(loader):
    """ Get the x,y split for a dataloader
    NOTE: this assumes the data is in the form (x,y) in the dataloader
    """
    temp_x,temp_y = [],[]
    for tx,ty in loader:
        temp_x.append(tx)
        temp_y.append(ty)
    return torch.cat(temp_x), torch.cat(temp_y)

# Get dataloader from sets of inidces
def get_dataloader(labels_idx, new_labels_idx, base_data, batch_size=8):
    """ This method takes in the old indices and the new indices requested by the model and
    generates a dataloader based on the union of the set of the two arrays of indices."""
    all_labels_idx = np.append(labels_idx, new_labels_idx)
    new_sampler = SubsetRandomSampler(all_labels_idx)
    new_loader = torch.utils.data.DataLoader(dataset=base_data, batch_size=batch_size, sampler=new_sampler)
    return new_loader

# Accuracy
def accuracy(model,x,y):
    """ Get classification accuracy """
    probs = model(Variable(x))
    _,ypred = torch.max(probs,1)
    acc = (ypred.data.numpy()==y.numpy()).sum()/len(y)
    return acc

def get_requested_points(model, unlab_loader, policy, num_points=16):
    """ This function gets the number of points requested based on the function
    "policy" that is passed. "policy" can be any function to test.
    ---------
    Args: model; any function that produces a valid array of length unlab_loader, but
            normally a torch model.
          unlab_loader; a torch DataLoader or other datatype of not using torch functionality
          policy; a passed fucntion that decides which points are most important for labeling
          num_points; int, the number of points indices we will return asking for labels.
    ---------
    Returns: array of length "num_points" that represent the indices of the points
                to be labeled.
    """
    unlab_preds = model(unlab_loader)
    return policy(unlab_preds, num_points)
