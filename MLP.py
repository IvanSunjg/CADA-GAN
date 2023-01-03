import torch
import torch.nn as nn
import torch.nn.functional as F

# Four-Layer network as described in GANkin, with additional dropout
class FourLayerNet(nn.Module):
    def __init__(self):
        super(FourLayerNet, self).__init__()
        self.fc1 = nn.Linear(2*16*512, 16*512)
        self.fc2 = nn.Linear(16*512, 16*512)
        self.unflatten = nn.Unflatten(1, torch.Size([16,512]))
        self.dropout = nn.Dropout(0.25)

    # x represents our data
    def forward(self, x):
        y = torch.flatten(x, 1)
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        out = self.unflatten(y)
        
        return out
