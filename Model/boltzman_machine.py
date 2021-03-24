import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as I
from torch.nn import Sigmoid
from torch.autograd import Variable

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
