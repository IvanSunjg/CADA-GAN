import torch
import torch.nn as nn
import torch.nn.functional as F

# Four-Layer network as described in GANkin but adapted to the use of StyleGAN, with additional dropout
class FourLayerNet(nn.Module):
    def __init__(self):
        super(FourLayerNet, self).__init__()
        self.fc1 = nn.Linear(18*512*2, 18*512*2)
        self.fc2 = nn.Linear(18*512*2, 18*512*2)
        self.fc3 = nn.Linear(18*512*2, 18*512*2)
        self.fc4 = nn.Linear(18*512*2, 18*512)
        self.dropout = nn.Dropout(0.25)

    # x represents our data
    def forward(self, x):
        x = nn.Flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x
