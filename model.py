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


    def save(self, file_name='linear_model.pth'):
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='linear_model.pth'):
        self.load_state_dict(torch.load(file_name))
        
class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        # x shape: (batch_size, sequence_length, input_size)
        if hidden is None:
            h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        else:
            h0, c0 = hidden

        out, (hn, cn) = self.lstm(x, (h0, c0))
        # On utilise la dernière sortie de la séquence
        out = self.lin(out[:, -1, :])
        return out, (hn, cn)

    def save(self, file_name='lstm_model.pth'):
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='lstm_model.pth'):
        self.load_state_dict(torch.load(file_name))
        
class PPOModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PPOModel, self).__init__()
        # Couche commune
        self.fc_common = nn.Linear(input_size, hidden_size)
        # Tête de la politique (acteur)
        self.fc_policy = nn.Linear(hidden_size, output_size)
        # Tête de la valeur (critique)
        self.fc_value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc_common(x))
        policy_logits = self.fc_policy(x)
        value = self.fc_value(x)
        return policy_logits, value
    
    
    def save(self, file_name='ppo_model.pth'):
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='ppo_model.pth'):
        self.load_state_dict(torch.load(file_name))