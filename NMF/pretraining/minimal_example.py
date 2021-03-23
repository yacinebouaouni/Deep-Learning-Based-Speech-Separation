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




class DNN(nn.Module):

    def __init__(self,d):
        super(DNN, self).__init__()
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




dnn_model = DNN(d=257)
model_1 = RBM(257,100)
model_1.load_state_dict(torch.load("w_layer_1"))
model_2 = RBM(100,50)
model_2.load_state_dict(torch.load("w_layer_2"))
model_3 = RBM(50,200)
model_3.load_state_dict(torch.load("w_layer_3"))


with torch.no_grad():
    dnn_model.fc1.weight = model_1.feed.weight
    dnn_model.fc2.weight = model_2.feed.weight
    dnn_model.fc3.weight = model_3.feed.weight
    