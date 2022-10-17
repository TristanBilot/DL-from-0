import torch
import torch.nn as nn
import torch.nn.functional as fun
from torchsummary import summary


class model_summary(nn.Module):
    def __init__(self):
        super(model_summary, self).__init__()
        self.fc = nn.Linear(10, 50)
        self.fc1 = nn.Linear(50, 1)
        self.loss = nn.MSELoss()

    def forward(self, z):
        z = self.fc(z)
        z = self.fc1(z)
        z = self.loss(z, torch.randn_like(z))
        return z

device_name = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
modl = model_summary().to(device_name)
# summary(modl, (300, 32))


import numpy as np

a = torch.Tensor(np.random.randn(32, 10))
model = model_summary()
res = model.forward(a)
res.backward()

b = torch.Tensor(np.random.randn(60, 10))
model = model_summary()
res = model.forward(b)
res.backward()

