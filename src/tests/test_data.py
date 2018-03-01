import numpy as np
import torch
import data as d

def test_get_samplers():
    _,_,_,i_idx,u_idx = al._get_samplers(100,4,4,random_seed=112)
    np.random.seed(112)
    my_idx = list(range(100))
    val_idx = np.random.choice(my_idx,size=4, replace=False)
    new_idx = list(set(my_idx)-set(val_idx))
    i_comp = np.random.choice(new_idx, size=4, replace=False)
    u_comp = list(set(new_idx)-set(i_comp))
    assert(np.allclose(i_idx,i_comp))
    assert(np.allclose(u_idx,u_comp))

def test_get_dataloader():
    x = torch.Tensor(np.array([[1,2,3],[7,7,7],[4,5,6]]))
    y = torch.Tensor(np.array([1,2,5]))
    dset = torch.utils.data.dataset.TensorDataset(x,y)
    data_load,_ = al.get_dataloader([0],[2],dset,batch_size=1)
    truths = []
    for x_t,y_t in data_load:
        if x_t.equal(torch.Tensor([1,2,3]).view(1,-1)) and y_t.numpy()[0]==1:
            truths.append(True)
            continue
        if x_t.equal(torch.Tensor([4,5,6]).view(1,-1)) and y_t.numpy()[0]==5:
            truths.append(True)
            continue
        truths.append(False)
    assert(all(truths))
