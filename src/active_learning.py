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


# TODO make flag for CUDA 
class ExperiAL(object):
    def __init__(self, model, train_x, train_y, val_x, val_y, loss_func, optimizer):
        self.model = model
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
        self.loss_func = loss_func
        self.optimizer = optimizer

    def _train(self, x, y, epochs=10, batch_size=8, shuffle=True):
        losses,itrs = [],0
        tensor_dataset = torch.utils.data.dataset.TensorDataset(x, y)
        tr_loader = torch.utils.data.DataLoader(dataset=tensor_dataset, batch_size=batch_size, shuffle=shuffle)

        for epoch in range(epochs):
            for i,(batch_x,batch_y) in enumerate(tr_loader):
                batch_x = Variable(batch_x)
                batch_y = Variable(batch_y)

                y_pred = self.model(batch_x)
                loss = self.loss_func(y_pred, batch_y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                itrs+=1
                losses.append(loss.data.numpy()[0])

        return list(range(itrs)), losses

    def active_learn(self, policy, meta_epochs=10, epochs_per_train=10, npoints=20, batch_size=8, random_seed=832):
        """ Active learning based on a specified policy. """
        total_acc = []
        unlab_x,unlab_y,lab_x,lab_y = get_uniform_split(self.train_x, self.train_y, n=npoints, random_seed=random_seed)

        for e in range(meta_epochs):
            # Train the model
            itr, losses = self._train(lab_x, lab_y, epochs=epochs_per_train, batch_size=batch_size)

            # Get the next points to label
            unlab_x,unlab_y, addtl_x, addtl_y = self.get_req_points(unlab_x, unlab_y, policy=policy, n=npoints, random_seed=random_seed)
            lab_x, lab_y = torch.cat([lab_x, addtl_x]), torch.cat([lab_y, addtl_y])

            # Get accuracy of the model
            total_acc.append(accuracy(self.model, self.val_x, self.val_y))

        return list(range(meta_epochs)), total_acc

    def get_req_points(self, unlab_x, unlab_y, policy, n, random_seed=13):
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
        if policy!='random':
            pred_y = self.model.forward(Variable(unlab_x))
            if policy=='boundary':
                fn = boundary_policy
            elif policy=='uniform':
                fn = uniform_policy
            elif policy=='max_entropy':
                fn = max_entropy_policy
            elif policy=='conf':
                fn = least_confidence_policy
            idxs = fn(pred_y, n=n)
            new_u_x, new_u_y, add_x, add_y = get_idx_split(unlab_x, unlab_y, idxs)
        else:
            new_u_x, new_u_y, add_x, add_y = get_dataset_split((unlab_x,unlab_y), other_size=n, random_seed=random_seed)
        return new_u_x, new_u_y, add_x, add_y

#################### POLICIES ###########################

def boundary_policy(pred_y, n):
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
    maxes, _ = torch.max(pred_y,dim=1)
    centered_around_max = pred_y.data.sub(maxes.data.view(-1,1).expand_as(pred_y.data))
    closest_col_to_zero = torch.sort(centered_around_max,dim=1)[0][:,-2]
    diffs_closest_to_zero = n_argmax(closest_col_to_zero, size=n)
    return diffs_closest_to_zero

# TODO docs
def max_entropy_policy(pred_y, n):
    """ docs"""
    probs = torch.exp(pred_y.data)
    prob_logprob = probs * pred_y.data
    max_ent = -torch.sum(prob_logprob, dim=1)
    max_ent_idxs = n_argmax(max_ent, size=n)
    return max_ent_idxs

def least_confidence_policy(pred_y, n):
    maxes = torch.max(pred_y.data,1)[0]
    least_conf = 1.0-maxes
    least_conf_idx = n_argmax(least_conf, size=n)
    return least_conf_idx

# TODO make this efficient - sampling without replacement to fill the quota, and then go random
def uniform_policy(pred_y, n):
    cut = n%10
    times = n//10
    _,preds = torch.max(pred_y,dim=1)
    output = []
    for res in range(10): # number of classes
        pred_idx = [idx for idx, elm in enumerate(preds) if elm.data.numpy()[0] == res]
        if len(pred_idx) < (times+1):
            if res<cut:
                output.extend(np.random.choice(range(len(preds)),size=times+1))
            else:
                output.extend(np.random.choice(range(len(preds)),size=times))
            continue
        if res<cut:
            output.extend(np.random.choice(pred_idx, size=times+1))
        else:
            output.extend(np.random.choice(pred_idx, size=times))
    return np.array(output)

#############################################################

# TODO: move to utils
def n_argmax(a,size):
    """ Find the n highest argmaxes of a 1D array or torch FloatTensor. """
    if type(a) == torch.FloatTensor:
        a = a.numpy()
    else:
        a = np.array(a)

    if len(a.shape)!=1:
        raise ValueError('Only 1D input supported.')

    return a.argsort()[-size:][::-1]

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

# Get the x,y seperation TODO move to utils
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

def get_idx_split(data_x, data_y, idx):
    x1_tensor = torch.cat([data_x[i].view(-1,28,28) for i in idx])
    y1_tensor = torch.LongTensor([data_y[i] for i in idx])
    x2_tensor = torch.cat([data_x[i].view(-1,28,28) for i in range(len(data_y)) if i not in idx])
    y2_tensor = torch.LongTensor([data_y[i] for i in range(len(data_y)) if i not in idx])
    return x2_tensor, y2_tensor, x1_tensor, y1_tensor

def get_uniform_split(train_x, train_y, n, random_seed=1823):
    np.random.seed(random_seed)
    cut = n%10
    times = n//10
    output = []
    for res in range(10): # number of classes
        y_idx = [idx for idx, elm in enumerate(train_y) if elm == res]
        if res<cut:
            output.extend(np.random.choice(y_idx, size=times+1))
        else:
            output.extend(np.random.choice(y_idx, size=times))
    return get_idx_split(train_x, train_y, output)

def get_dataset_split(train_set, other_size=10000, random_seed=1992):
    """ DOCS"""
    np.random.seed(random_seed)
    if isinstance(train_set, tuple):
        msk = np.random.choice(range(len(train_set[1])),size=other_size, replace=False)
        x1_tensor = torch.cat([train_set[0][i].view(-1,28,28) for i in msk])
        y1_tensor = torch.LongTensor([train_set[1][i] for i in msk])
        x2_tensor = torch.cat([train_set[0][i].view(-1,28,28) for i in range(len(train_set[1])) if i not in msk])
        y2_tensor = torch.LongTensor([train_set[1][i] for i in range(len(train_set[1])) if i not in msk])
    else:
        msk = np.random.choice(range(len(train_set)),size=other_size, replace=False)
        x1_tensor = torch.cat([train_set[i][0] for i in msk])
        y1_tensor = torch.LongTensor([train_set[i][1] for i in msk])
        x2_tensor = torch.cat([train_set[i][0] for i in range(len(train_set)) if i not in msk])
        y2_tensor = torch.LongTensor([train_set[i][1] for i in range(len(train_set)) if i not in msk])
    return x2_tensor, y2_tensor, x1_tensor, y1_tensor
