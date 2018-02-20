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
    truths = []
    for x_t,y_t in data_load:
        if x_t.equal(torch.Tensor([1,2,3]).view(1,-1) and y_t.numpy()[0]==1:
            truths.append(True)
            continue
        if x_t.equal(torch.Tensor([4,5,6]).view(1,-1) and y_t.numpy()[0]==5:
            truths.append(True)
            continue
        truths.append(False)
    assert(all(truths))

def test_accuracy():
    x = torch.Tensor(np.array([[1,2,3],[7,7,7],[4,5,6]]))
    y = torch.Tensor(np.array([1,5,5]))
    model = lambda x: torch.sum(x,axis=1)
    probs = model(Variable(x))
    _,ypred = torch.max(probs,1)
    acc = (ypred.data.numpy()==y.numpy()).sum()/len(y)

def test_get_samplers():
    pass
