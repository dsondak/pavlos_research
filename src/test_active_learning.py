import numpy as np
import torch
import active_learning as al

def test_get_requested_points():
    mod = np.vectorize(lambda x: x**2+1)
    unlab_loader = [i/2 for i in range(1,5)]
    policy = lambda x,n: x+n
    result = al.get_requested_points(mod, unlab_loader, policy)
    assert(np.allclose(result,np.array([17.25,18.,19.25,21.])))

def test_get_xy_split():
    gen = ((x,x**2 for x in range(1,5))
    x,y = al.get_xy_split()
    assert(np.allclose(x,np.array([1,2,3,4])))
    assert(np.allclose(y,np.array([1,4,9,16])))

def test_get_dataloader():
    x = torch.Tensor(np.array([[1,2,3],[7,7,7],[4,5,6]]))
    y = torch.Tensor(np.array([1,2,5]))
    dset = torch.utils.data.dataset.TensorDataset(x,y)
    data_load = al.get_dataloader([0],[2],dset,batch_size=1)
    for i in data_load:
        pass 

def test_accuracy():
    pass

def test_get_samplers():
    pass
