import torch
from torch.utils.data import Dataset, DataLoader







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