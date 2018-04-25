import os
import sys
# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)
import src.active_learning as al
import src.data as d 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

# Get raw dataset - USPS
percent_test = 0.3
usps_batch = 64
usps_set = d.get_usps('../notebooks/usps/usps_all.mat', size=(28,28))
usps_x, usps_y, usps_test_x, usps_test_y = al.get_dataset_split(usps_set,int(len(usps_set)*percent_test))
usps_test_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(usps_test_x, usps_test_y), \
                                               batch_size=len(usps_test_y),shuffle=False)
usps_train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(usps_x, usps_y), \
                                               batch_size=usps_batch,shuffle=True)

# Define the CNN model
class CNN(nn.Module):
    """ CNN for MNIST """
    def __init__(self):
      super(CNN,self).__init__()
      self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
      self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
      self.conv2_drop = nn.Dropout2d()
      self.fc1 = nn.Linear(320, 50)
      self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
      x = x.view(-1,1,28,28)
      x = F.relu(F.max_pool2d(self.conv1(x), 2))
      x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
      x = x.view(-1, 320)
      x = F.relu(self.fc1(x))
      x = self.fc2(x)
      return F.log_softmax(x,dim=1)

# train simple learner on USPS and use as transfer option
usps_epochs = 100
use_cuda = torch.cuda.is_available()
usps_model = CNN().cuda() if use_cuda else CNN()
usps_opt = optim.Adam(usps_model.parameters(), lr=0.01)
usps_loss = nn.NLLLoss()

usps_losses,n_itr = [],0
for e in tqdm(range(usps_epochs)):
    for batch_x, batch_y in usps_train_loader:
        batch_x = Variable(batch_x.cuda()) if use_cuda else Variable(batch_x)
        batch_y = Variable(batch_y.cuda()) if use_cuda else Variable(batch_y)

        result = usps_model(batch_x)
        loss = usps_loss(result, batch_y)
        usps_opt.zero_grad()
        loss.backward()
        usps_opt.step()

        n_itr+=1
        usps_losses.append(loss)

if use_cuda:
    usps_losses = [itm.data.cpu().numpy()[0] for itm in usps_losses]
    torch.save(usps_model.cpu(), 'usps_model.pt')
else:
    usps_losses = [itm.data.numpy()[0] for itm in usps_losses]
    torch.save(usps_model, 'usps_model.pt')

# Analysis of model 
plt.plot(range(n_itr), usps_losses, alpha=0.7)
plt.title('Loss of CNN training on usps handwritten images.')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()

print('Training accuracy:',al.accuracy(usps_model, usps_x, usps_y, use_cuda))
print('Test accuracy:',al.accuracy(usps_model, usps_test_x, usps_test_y, use_cuda))
