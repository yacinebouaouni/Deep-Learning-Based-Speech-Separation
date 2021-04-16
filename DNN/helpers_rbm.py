from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from scipy import signal
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
from numpy.linalg import inv
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
from torch.nn import Sigmoid


from torch import transpose
from sklearn.neural_network import BernoulliRBM
import seaborn as sns
import warnings
warnings.simplefilter('ignore')
import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt

class AudioDataset(Dataset):
    """Sources dataset."""

    def __init__(self, Y, s1,s2):
        
        self.s1 = torch.tensor(s1)
        self.s2 = torch.tensor(s2)
        self.Y = Y
        
    def __len__(self):
        return self.s1.size()[0]

    def __getitem__(self, idx):
        
        return {'Y':self.Y[idx,:],'s1': self.s1[idx,:], 's2': self.s2[idx,:]}
class RBM(nn.Module):

    def __init__(self,
                 visible_size,
                 hidden_size):
        super(RBM, self).__init__()
        self.input_layer = visible_size
        self.output_layer = hidden_size
        self.dropout = torch.nn.Dropout(0.005)
        
        self.feed = nn.Linear(visible_size, hidden_size,bias=True)  

    def forward(self, x):
        h = Sigmoid()(self.dropout(self.feed(x)))       # x -> h 

        x_reconstructed = (self.dropout(torch.matmul(h,self.feed.weight)))
        
        h_reconstructed = Sigmoid()(self.dropout(self.feed(x_reconstructed)))# h-> x'



        return h, x_reconstructed,h_reconstructed



def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def training_rbm(model,               ### Train LOOP OF RBM
                 trainloader,
                 learning_rate = 0.001,
                 n_epochs=200,
                 l2_penalty=0.01):

  
  loss_s1 = []
  loss_s2 = []
  ## weights initialization
  model.apply(init_weights)

  criterion = torch.nn.MSELoss()
  optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,weight_decay=l2_penalty)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr = 0.09e-3,
                                                         mode = 'min',
                                                         factor=0.9999,
                                                         verbose=True
                                                         ,patience=11)

  print('Begin Training.....')

  for epoch in range(n_epochs):  # loop over the dataset multiple times

      running_loss = 0.0
      for i,data in enumerate(trainloader):
          a = model.feed.weight.t()  


          model.zero_grad()     

          h1,x_reconstructed_1,h_reconstructed_1 = model(data['s1'])     ##  calcul x_reconstructed
          loss_1 = -torch.sum(a*(torch.matmul(data['s1'].t(),h1) - torch.matmul(x_reconstructed_1.t(),h_reconstructed_1)))/(data['s1'].shape[0]*data['s1'].shape[1])
          #loss_1 = torch.sum(KL_divergence(data['s1'],x_reconstructed_1))
          loss_1.backward()
          optimizer.step()    ## W -= W - lr*dL/dW

          mse_1 = criterion(data['s1'],x_reconstructed_1)
          loss_s1.append(mse_1)
          model.zero_grad()     


          h2,x_reconstructed_2,h_reconstructed_2 = model(data['s2'])     ##  calcul x_reconstructed
          loss_2 = -torch.sum(a*(torch.matmul(data['s2'].t(),h2) - torch.matmul(x_reconstructed_2.t(),h_reconstructed_2)))/(data['s2'].shape[0]*data['s2'].shape[1])
          #loss_2 = torch.sum(KL_divergence(data['s2'],x_reconstructed_2))

          loss_2.backward()
          optimizer.step()
          mse_2 = criterion(data['s2'],x_reconstructed_2)
          loss_s2.append(mse_2)
          #loss = torch.add(loss_1)#,loss_2)
                ## W* (xh_t - x'_t.  h'_t)
              ## (xh_t - x'_t. h'_t)
      scheduler.step(epoch)
      if epoch%10 == 0:
        print("epochs {} MSE Loss {}".format(epoch,torch.mean(criterion(data['s1'],x_reconstructed_1) + criterion(data['s2'],x_reconstructed_2) )))
  print('Finished Training')

  
   
