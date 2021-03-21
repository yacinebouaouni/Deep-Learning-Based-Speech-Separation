import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as I
from torch.nn import Sigmoid 
class RBM(nn.Module):

    def __init__(self,
                 visible_size,
                 hidden_size):
        super(RBM, self).__init__()
        self.input_layer = visible_size
        self.output_layer = hidden_size
        
        self.feed = nn.Linear(visible_size, hidden_size)  


    def forward(self, x):
        h = Sigmoid()(self.feed(x))        # x -> h 

        x_reconstructed = torch.matmul(h,self.feed.weight)
        
        h_reconstructed = Sigmoid()(self.feed(x_reconstructed))# h-> x'

        return h,x_reconstructed,h_reconstructed

data = torch.rand(5210,257)
model = Net(data.shape[1], 200)
model(data)