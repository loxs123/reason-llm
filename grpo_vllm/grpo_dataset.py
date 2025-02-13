import torch.utils.data as data
import json
import time
import random

class GRPODataset(data.Dataset):
    def __init__(self, filename):
        super(GRPODataset, self).__init__()
        with open(filename) as f:
            data = json.load(f)
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        return item
