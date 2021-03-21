import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I



class RBM(nn.Module):

    def __init__(self,
                 input_layer,
                 output_layer):
        super(RBM, self).__init__()
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.sigmoid = torch.nn.Sigmoid()
        
        self.feed = nn.Linear(input_layer, output_layer)  


    def forward(self, x):
        h = self.sigmoid(self.feed(x))        # x -> h 

        x_reconstructed = torch.matmul(h,self.feed.weight)
        
        h_reconstructed = self.sigmoid(self.feed(x_reconstructed))# h-> x'

        return h,x_reconstructed,h_reconstructed

data = torch.rand(5210,257)
model = Net(data.shape[1], 200)
model(data)