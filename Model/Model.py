import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I




class Net(nn.Module):

    def __init__(self,d):
        super(Net, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.fc1 = nn.Linear(d, 100)  # d is dimension of the input.
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 200)
        self.fc4 = nn.Linear(200,2)
        

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        x = self.sigmoid(self.fc4(x))


        return x
    
    
    
class Net_r(nn.Module):

    def __init__(self,d):
        super(Net, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.fc1 = nn.Linear(d, 100)  # d is dimension of the input.
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 200)
        self.fc4 = nn.Linear(200,2)
        

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        x = self.sigmoid(self.fc4(x))


        return x