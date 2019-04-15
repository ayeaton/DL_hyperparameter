import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

model = models.inception_v3(pretrained=False)
model.fc = nn.Linear(2048, 1)
model.AuxLogits.fc = nn.Linear(768, 1)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.inception = model

    def forward(self, x):
        x = self.inception(x)
        x = nn.LogSoftmax(x)
        return(x)
