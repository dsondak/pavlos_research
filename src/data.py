# Imports
import numpy as np
import pandas as pd
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from scipy.io import loadmat
from PIL import Image

def _get_samplers(total_len,init_size,val_size=10000, random_seed=1492):
    """ This function gets the samplers for initialization.  The purpose is so
    the user can specify the total number of data points, and have that number broken
    up into a validation set of indices, and a "init_size" specified initial
    training set.  Foor the purposes of this problem the training set is usually
    small and the remaining "unlabeled" dataset is rather large, for clear changes
    using the active learning policies.
    ---------
    Args: total_len; int, total number of examples in dataset in question.
          init_size; int, desired initial number of points in the training set.
          val_size; int, desired number of points in the validation set.
          random_seed; int, random seed value for sampling.
    ---------
    Returns: train_sampler; a SubsetRandomSampler of length init_size
             unlabeled_sampler; a SubsetRandomSampler of length total_len-init_size-val_size
             val_sampler; a SubsetRandomSampler of length val_size
             init_labels_idx; list of the indices of the training samples
             unlabeled_idx; list of the inidices of the "unlabeled" samples.
    """
    if random_seed:
        np.random.seed(random_seed)
    # Get Indices for validation and for initial random training sample
    idxs = list(range(total_len))
    val_idx = np.random.choice(idxs,size=val_size, replace=False)
    train_idx = list(set(idxs)-set(val_idx))
    init_labels_idx = np.random.choice(train_idx,size=init_size, replace=False)
    unlabeled_idx = list(set(train_idx)-set(init_labels_idx))

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

# Get dataloader from sets of inidces
def get_dataloader(labels_idx, new_labels_idx, base_data, batch_size=8):
    """ This method takes in the old indices and the new indices requested by the model and
    generates a dataloader based on the union of the set of the two arrays of indices.
    -------
    Args: labels_idx; array or list of original indices of training points
          new_labels_idx; array or list of new points to get labels for (chosen by policy)
          base_data; the dataset (a torch DataSet object)
          batch_size; int, the desired batch size of the new DataLoader
    -------
    Returns: torch DataLoader object with the old indices of data and the new indices.
             numpy array with the aggreated indices of the two passed lists
    """
    all_labels_idx = np.append(labels_idx, new_labels_idx)
    new_sampler = SubsetRandomSampler(all_labels_idx)
    new_loader = torch.utils.data.DataLoader(dataset=base_data, batch_size=batch_size, sampler=new_sampler)
    return new_loader, all_labels_idx

def get_usps(file_path, size=(28,28)):
    """ DOCS """
    usps = loadmat(file_path)
    assert(usps['data'].shape==(256,1100,10))

    data,resp = [],[]
    for digit in range(10):
        for elm in range(1100):
            img = Image.fromarray(usps['data'][:,elm,digit].reshape(16,16))
            data.append(transforms.ToTensor()(transforms.Resize(size)(img)))
            r = digit+1 if digit != 9 else 0
            resp.append(r)

    usps_x = torch.cat(data).view(-1,1,*size)
    usps_y = torch.LongTensor(resp)
    return torch.utils.data.TensorDataset(usps_x, usps_y)
