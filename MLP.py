import torch
import torch.nn as nn
import torch.nn.functional as F

# Four-Layer network as described in GANkin, with additional dropout
class FourLayerNet(nn.Module):
    def __init__(self):
        super(FourLayerNet, self).__init__()
        self.fc1 = nn.Linear(2*18*512, 18*512)
        self.fc2 = nn.Linear(18*512, 18*512)
        self.fc3 = nn.Linear(18*512, 18*512)
        self.fc4 = nn.Linear(18*512, 18*512)

        self.dropout = nn.Dropout(0.25)

    # x represents our data
    def forward(self, x):
        y = torch.flatten(x, 1)
        y = self.fc1(y)
        y = F.relu(y)
        y = self.dropout(y)
        y = self.fc2(y)
        y = F.relu(y)
        y = self.dropout(y)
        y = self.fc3(y)
        y = F.relu(y)
        y = self.dropout(y)
        out = self.fc4(y)
        
        return out
