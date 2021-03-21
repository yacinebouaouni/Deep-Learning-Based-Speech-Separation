
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
class Net(nn.Module):

    def __init__(self,
                 input,
                 output):
        super(Net, self).__init__()
        self.input = input
        self.output = output
        self.sigmoid = torch.nn.Sigmoid()
        
        self.feed = nn.Linear(input, output)  


    def forward(self, x):
        h = self.sigmoid(self.feed(x))        # x -> h 

        x_reconstructed = torch.matmul(self.feed.weight.t(), h.t())
        
        h_reconstructed = self.sigmoid(self.feed(x_reconstructed.t()))# h-> x'




        return x_reconstructed, h , h_reconstructed

data = torch.rand(5210,257)
model = Net(input = data.shape[1],output = 200)
model(data)