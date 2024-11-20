# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LinQNet, self).__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x


    def save(self, file_name='model.pth'):
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model.pth'):
        self.load_state_dict(torch.load(file_name))