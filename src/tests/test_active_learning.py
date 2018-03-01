import numpy as np
import torch
import active_learning as al


## Global model for testing
# def test_get_req_points():
#     ux, uy = torch.FloatTensor([[3,4],[1,18]]), torch.LongTensor([0,1])
#     mod = lambda x: torch.pow(x,2)
#     policy = 'random'
#     result = al.get_req_points(ux, uy, policy, n=2, random_seed=13)
#     print(result)
#     assert(np.allclose(result.data.numpy(),np.array([16.25,17.,18.25,20.])))

def test_get_xy_split():
    gen = ((torch.Tensor([x]),torch.Tensor([x**2])) for x in range(1,5))
    x,y = al.get_xy_split(gen)
    assert(np.allclose(x,np.array([1,2,3,4])))
    assert(np.allclose(y,np.array([1,4,9,16])))

def test_accuracy():
    x = torch.autograd.Variable(torch.Tensor(np.array([[1,2,3],[7,7,7],[4,5,6]])))
    y = torch.Tensor(np.array([2,0,1]))
    mod = lambda x: torch.nn.functional.softmax(x,dim=1)
    _,preds = torch.max(mod(x),1)
    tf = np.mean(preds.data.numpy()==y.numpy().astype(int))
    pf = al.accuracy(mod,x.data,y)
    assert(np.allclose(tf,pf))

def test_boundary_policy():
    up = torch.autograd.variable.Variable(torch.Tensor([[1,2],[4,9]]))
    res = al.boundary_policy(up,n=1)
    assert(np.allclose(res,[0]))

def test_n_argmax():
    a = np.array([4,3,2,1,7])
    assert(np.allclose(al.n_argmax(a,size=2),[4,0]))
