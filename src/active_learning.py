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

# TODO debug for potential cuda probs
class ExperiAL(object):
    """ Active Learning Experiment object """
    def __init__(self, model, train_x, train_y, val_x, val_y, loss_func, optimizer, params='default',random_seed=128):
        self.use_cuda = torch.cuda.is_available()
        self.model = model.cuda() if self.use_cuda else model
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
        self.loss_func = loss_func.cuda() if self.use_cuda else loss_func
        self.optimizer = optimizer
        if params=='default':
            self.set_params(meta_epochs=1, npoints=20, batch_size=10, epochs_per_train=5, shuffle=True)
        elif isinstance(params, dict):
            self.set_params(meta_epochs=params['meta_epochs'], npoints=params['npoints'], \
                            batch_size=params['batch_size'], epochs_per_train=params['epochs_per_train'])
        unlab_x,unlab_y,lab_x,lab_y = get_uniform_split(self.train_x, self.train_y, n=self.npoints, random_seed=random_seed)
        self.unlab_x = unlab_x.cuda() if self.use_cuda else unlab_x
        self.unlab_y = unlab_y.cuda() if self.use_cuda else unlab_y
        self.lab_x = lab_x.cuda() if self.use_cuda else lab_x
        self.lab_y = lab_y.cuda() if self.use_cuda else lab_y


    def set_params(self, **kwargs):
        """ Set active learning parameters """
        keys = kwargs.keys()
        if 'batch_size' in keys:
            self.batch_size = kwargs['batch_size']
        if 'epochs_per_train' in keys:
            self.ept = kwargs['epochs_per_train']
        if 'npoints' in keys:
            self.npoints = kwargs['npoints']
        if 'meta_epochs' in keys:
            self.meta_epochs = kwargs['meta_epochs']
        if 'shuffle' in keys:
            self.shuffle = kwargs['shuffle']

    def _train(self, x, y):
        """ Function to train the model on specified data for a number of epochs
        --------
        Args: x; torch FloatTensor to train on
              y; torch LongTensor to train on
        --------
        Returns: list of iterations, losses per iteration
        """
        losses,itrs = [],0
        tensor_dataset = torch.utils.data.dataset.TensorDataset(x, y)
        tr_loader = torch.utils.data.DataLoader(dataset=tensor_dataset, batch_size=self.batch_size,
                                                shuffle=self.shuffle, pin_memory=self.use_cuda)
        for epoch in range(self.ept):
            for i,(batch_x,batch_y) in enumerate(tr_loader):

                y_pred = self.model(Variable(batch_x))
                loss = self.loss_func(y_pred, Variable(batch_y))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                itrs+=1
                next_loss = loss.data.cpu().numpy()[0] if self.use_cuda else loss.data.numpy()[0]
                losses.append(next_loss)

        return list(range(itrs)), losses

    def active_learn(self, policy, random_seed=832):
        """ Active learning based on a specified policy.
        -------
        Args: policy; str, specifies which policy to perform
            .
            .
            .
        -------
        Returns: meta_epochs as a list, validation accuracy per meta epoch
        """
        total_acc = []
        for e in range(self.meta_epochs):
            # Train the model
            itr, losses = self._train(self.lab_x, self.lab_y)

            # Get the next points to label
            self.unlab_x, self.unlab_y, addtl_x, addtl_y = self.get_req_points(policy=policy, random_seed=random_seed)
            self.lab_x, self.lab_y = torch.cat([self.lab_x, addtl_x]), torch.cat([self.lab_y, addtl_y])

            # Get accuracy of the model
            total_acc.append(accuracy(self.model, self.val_x, self.val_y, self.use_cuda))

        return list(range(self.meta_epochs)), total_acc

    def get_req_points(self, policy, random_seed=13):
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
            pred_y = self.model(Variable(self.unlab_x))
            if policy=='boundary':
                fn = boundary_policy
            elif policy=='uniform':
                fn = uniform_policy
            elif policy=='max_entropy':
                fn = max_entropy_policy
            elif policy=='conf':
                fn = least_confidence_policy
            idxs = fn(pred_y, n=self.npoints, use_cuda=self.use_cuda)
            new_u_x, new_u_y, add_x, add_y = get_idx_split(self.unlab_x, self.unlab_y, idxs)
        else:
            new_u_x, new_u_y, add_x, add_y = get_dataset_split((self.unlab_x,self.unlab_y), other_size=self.npoints, random_seed=random_seed)
        return new_u_x, new_u_y, add_x, add_y

#################### POLICIES ###########################

def boundary_policy(pred_y, n, use_cuda=False):
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
    closest_to_zero = torch.sort(centered_around_max,dim=1)[0][:,-2]
    diffs_zero = n_argmax(closest_to_zero.cpu(), size=n) if use_cuda else n_argmax(closest_to_zero, size=n)
    return diffs_zero

def max_entropy_policy(pred_y, n, use_cuda=False):
    """ Take the maximum entropy of the resulting probabilities.
    NOTE: favors situations where we are generally confused about everything
    """
    probs = torch.exp(pred_y.data)
    prob_logprob = probs * pred_y.data
    max_ent = -torch.sum(prob_logprob, dim=1)
    max_ent_idxs = n_argmax(max_ent.cpu(), size=n) if use_cuda else n_argmax(max_ent, size=n)
    return max_ent_idxs

def least_confidence_policy(pred_y, n, use_cuda=False):
    """ Take the least confidence of the resulting probabilities.
    NOTE: favors situations where we are generally confused about everything
    """
    maxes = torch.max(pred_y.data,1)[0]
    least_conf = 1.0-maxes
    least_conf_idx = n_argmax(least_conf.cpu(), size=n) if use_cuda else n_argmax(least_conf, size=n)
    return least_conf_idx

def uniform_policy(pred_y, n, use_cuda=False):
    """ Sample uniformly from the predictions on the unlabeded points"""
    cut = n%10
    times = n//10
    output = []
    _,preds = torch.max(pred_y,dim=1)
    num_points, sampler = len(preds), np.array(range(len(preds)))
    mixed_idxs = np.random.choice(sampler, size=num_points, replace=False)
    class_counter = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,'rand':0}
    for mi in mixed_idxs:
        pred_class = preds[mi].data.cpu().numpy()[0] if use_cuda else preds[mi].data.numpy()[0]

        if class_counter[pred_class]<times:
            output.append(mi)
            class_counter[pred_class]+=1
        elif class_counter['rand']<cut:
            output.append(mi)
            class_counter['rand']+=1

        if sum(class_counter.values())==n:
            break

    return np.array(output)

#############################################################

# TODO: move to utils
def n_argmax(a,size):
    """ Find the n highest argmaxes of a 1D array or torch FloatTensor.
    -------
    Args: a; FloatTensor or numpy arrays
          size; int, the number of argmaxes you want
    -------
    Returns: numpy array
    """
    if type(a) == torch.FloatTensor:
        a = a.numpy()
    else:
        a = np.array(a)

    if len(a.shape)!=1:
        raise ValueError('Only 1D input supported.')

    return a.argsort()[-size:][::-1]

# Accuracy
def accuracy(model,x,y,use_cuda=False):
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
    if use_cuda:
        probs = model(Variable(x.cuda()))
        _,ypred = torch.max(probs,1)
        return (ypred.data.cpu().numpy()==y.cpu().numpy()).sum()/len(y)
    else:
        probs = model(Variable(x))
        _,ypred = torch.max(probs,1)
        return (ypred.data.numpy()==y.numpy()).sum()/len(y)
    # acc = (ypred.data.numpy()==y.numpy()).sum()/len(y)
    # return acc

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
    """ For some torch Tensors data_x and data_y return one set of tensors
    for every index in the list passes and another set that is the compliment of that.
    -------
    Args: data_x; torch Tensor, all x values
          data_y; torch Tensor, all y values
          idx; numpy array, the indexes we want to get a seperate torch tensor pair
            for. Also return the compliment so no data points are lost
    -------
    Returns: tuple; compliment tensor x, tensor y, index tensor x tensor y
    """
    x1_tensor = torch.cat([data_x[i].view(-1,28,28) for i in idx])
    y1_tensor = torch.LongTensor([data_y[i] for i in idx])
    x2_tensor = torch.cat([data_x[i].view(-1,28,28) for i in range(len(data_y)) if i not in idx])
    y2_tensor = torch.LongTensor([data_y[i] for i in range(len(data_y)) if i not in idx])
    return x2_tensor, y2_tensor, x1_tensor, y1_tensor

def get_uniform_split(train_x, train_y, n, random_seed=1823):
    """ Wrapper for get_idx_split that ensures we have a uniform initial sample
    based on the y labels
    --------
    Args: train_x; torch Tensor of all x
          train_y; torch Tensor for all y
          n; int, n the number of points to sample with a uniform dist on y
          random_seed; int, the random seed
    --------
    Returns: tuple; compliment tensor x, tensor y, index tensor x tensor y
    """
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
    """ Function to split either a Dataset torch object or a tuole of two torch
    Tensors randomly.
    ------
    Args: train_set; either a torch Dataset or a tuple of two torch Tensors, x,y respectively
          other_size; int, the size of the split (returned after the compliment)
          random_seed; int, the random seed
    ------
    Returns: tuple; compliment tensor x, tensor y, index tensor x tensor y
    """
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
