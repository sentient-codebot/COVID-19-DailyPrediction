import torch
import torch.nn as nn
from RIM import RIMCell

class RIMModel(nn.Module):
    def __init__(self, device, hidden_size, num_units, k, rnn_cell):
        super().__init__()
        self.rim_cell = RIMCell(device, 1, hidden_size, num_units, k, rnn_cell)
        self.device = device
        self.num_units = num_units
        self.hidden_size = hidden_size
        self.rnn_cell = rnn_cell
        self.out_linear = nn.Linear(hidden_size*num_units, 1) # output linear layer

    def forward(self, past):
        '''past (BATCH, p)'''
        past_split = torch.split(past, 1, 1)
        hs = torch.randn(past.shape[0], self.num_units, self.hidden_size).to(self.device)
        cs = None 
        if self.rnn_cell == 'LSTM':
            cs = torch.randn(past.shape[0], self.num_units, self.hidden_size).to(self.device)

        for past_step in past_split:
            hs, cs = self.rim_cell(past_step, hs, cs)
        pred = self.out_linear(hs.reshape(past.shape[0], -1))

        return pred

class MLPModel(nn.Module):
    '''
    one-layer MLP
    '''
    def __init__(self, device, p, hidden_size):
        super().__init__()
        self.hidden_linear = nn.Linear(p, hidden_size)
        self.out_linear = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, past):
        hidden = self.hidden_linear(past)
        hidden = self.relu(hidden)
        out = self.out_linear(hidden)
        out = self.relu(out)

        return out