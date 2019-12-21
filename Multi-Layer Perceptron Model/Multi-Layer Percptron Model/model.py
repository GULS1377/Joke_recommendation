import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class JokeRsNN(nn.Module):

    def __init__(self):
        super(JokeRsNN, self).__init__()

        self.fc_u = nn.Linear(32, 32)
        self.fc_j = nn.Linear(32, 32)

        self.fc_final = nn.Linear(64, 1)

    def forward(self, u, j):
        x_u = F.relu(self.fc_u(u))
        x_j = F.relu(self.fc_j(j))

        x_user = F.tanh(self.fc_u(x_u))
        x_joke = F.tanh(self.fc_j(x_j))

        output = self.fc_final(torch.cat((x_user, x_joke), 1))
        return output[:, 0]
