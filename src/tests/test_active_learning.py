import numpy as np
import torch
import active_learning as al

def test_get_requested_points():
    mod = lambda x: torch.pow(x,2)
    unlab_loader = [(torch.Tensor([i/2 for i in range(1,5)]),torch.Tensor([1,1,1,1]))]
    policy = lambda x,t,n: x+n
    result = al.get_requested_points(mod, unlab_loader, [1,2,3,4], policy)
    assert(np.allclose(result.data.numpy(),np.array([16.25,17.,18.25,20.])))

def test_get_xy_split():
    gen = ((torch.Tensor([x]),torch.Tensor([x**2])) for x in range(1,5))
    x,y = al.get_xy_split(gen)
    assert(np.allclose(x,np.array([1,2,3,4])))
    assert(np.allclose(y,np.array([1,4,9,16])))

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

def test_accuracy():
    x = torch.autograd.Variable(torch.Tensor(np.array([[1,2,3],[7,7,7],[4,5,6]])))
    y = torch.Tensor(np.array([2,0,1]))
    mod = lambda x: torch.nn.functional.softmax(x,dim=1)
    _,preds = torch.max(mod(x),1)
    tf = np.mean(preds.data.numpy()==y.numpy().astype(int))
    pf = al.accuracy(mod,x.data,y)
    assert(np.allclose(tf,pf))

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

def test_boundary_proximity():
    up = torch.autograd.variable.Variable(torch.Tensor([[1,2],[4,9]]))
    res = al.boundary_proximity(up,num_points=1)
    assert(np.allclose(res,[0]))

def test_n_argmax():
    a = np.array([4,3,2,1,7])
    assert(np.allclose(al.n_argmax(a,size=2),[4,0]))
