# Imports
import numpy as np
# import pandas as pd
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
# import pyro


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

# Get the x,y seperation
def get_xy_split(loader):
    """ Get the x,y split for a dataloader
    NOTE: this assumes the data is in the form (x,y) in the dataloader
    -------
    Args: DataLoader object ... see NOTE above
    -------
    Returns: tuple of Tensor x, Tensor y
    """
    temp_x,temp_y = [],[]
    for tx,ty in loader:
        temp_x.append(tx)
        temp_y.append(ty)
    return torch.cat(temp_x), torch.cat(temp_y)

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

# Accuracy
def accuracy(model,x,y):
    """ Get classification accuracy for a torch model (or any function that can take
    torch Variables as input).
    ---------
    Args: model; a torch model, or any function that can take torch Variable inputs
          x; torch Tensor object to be made into a Variable before running through
                model.
          y; torch Tensor with outputs of classification for comparison with model(x)
    ---------
    Returns: float; the classification accuracy of the model.
    """
    probs = model(Variable(x))
    _,ypred = torch.max(probs,1)
    acc = (ypred.data.numpy()==y.numpy()).sum()/len(y)
    return acc

def get_requested_points(model, unlab_loader, unlab_idx, policy, num_points=16):
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
    if len(unlab_loader) != 1:
        raise ValueError('The unlabeled loader must be one batch of the \
                            whole size of the unlabeled data.')
    for x,y in unlab_loader:
        unlab_preds = model(Variable(x))
    return policy(unlab_preds, unlab_idx, num_points)

def boundary_proximity(unlab_preds,num_points):
    """ Takes in a torch Variable predictions and outputs the difference in the largest output of the
    model and the next closest.  The smaller this number the closer the prediction is to the boundary.
    This function works by finding the max of the input predictions and then subtracting that value
    from its respective column. This leaves 0 as the maximum and the argmax w/o conisidering 0 is the
    difference between the highest and second closest value.  We return the argmaxes of the "num_points"
    closest values.
    --------
    Args: unlab_preds; torch Variable with the predictions (usually log probs or probs) of the model
          num_points; int the number of points to return
    --------
    Returns: numpy array of argmaxes of the points closest to the boundary.
    """
    maxes,_ = torch.max(unlab_preds,dim=1)
    centered_around_max = unlab_preds.data.sub(maxes.data.view(-1,1).expand_as(unlab_preds.data))
    closest_col_to_zero = torch.sort(centered_around_max,dim=1)[0][:,-2]
    diffs_closest_to_zero = n_argmax(closest_col_to_zero, size=num_points)
    return diffs_closest_to_zero

def n_argmax(a,size):
    """ Find the n highest argmaxes of a 1D array or torch FloatTensor. """
    if type(a) == torch.FloatTensor:
        a = a.numpy()
    else:
        a = np.array(a)

    if len(a.shape)!=1:
        raise ValueError('Only 1D input supported.')

    return a.argsort()[-size:][::-1]
