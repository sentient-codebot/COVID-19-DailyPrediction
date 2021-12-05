import torch
import torch.nn as nn
from RIM import RIMCell

class RIMModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = torch.device('cuda') if args.cuda else torch.device('cpu')
        self.cuda = True if args.cuda else False
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.num_units = args.num_units
        self.kA = args.kA
        self.rnn_cell = args.rnn_cell
        self.output_size = args.output_size
        self.RIMModel = RIMCell(self.device, self.input_size, self.hidden_size, self.num_units, self.kA, self.rnn_cell)

        self.Output = nn.Linear(self.hidden_size * self.num_units, self.output_size) # NOTE: really? use all hidden_states or only activated?

    def forward(self, seq_past):
        '''
        seq_past (BATCHSIZE, SEGMENT_LENGTH)
        '''
        if self.cuda:
            seq_past = seq_past.to(self.device)

        hs = torch.randn(seq_past.size(0), self.num_units, self.hidden_size).to(self.device)
        cs = None
        if self.rnn_cell == 'LSTM':
            cs = torch.randn(seq_past.size(0), self.num_units, self.hidden_size).to(self.device)
        seq_split = torch.split(seq_past, self.input_size, 1)
        for seq_entry in seq_split:
            hs, cs, _ = self.RIMModel(seq_entry, hs, cs)
        predicted = self.Output(hs.reshape(seq_past.size(0),-1))

        return predicted

class MLPModel(nn.Module):
    '''
    one-layer MLP
    '''
    def __init__(self, device, p, hidden_size):
        super().__init__()
        self.hidden_linear = nn.Linear(p, hidden_size)
        self.out_linear = nn.Linear(hidden_size, 1)
        self.activation = nn.ReLU()

    def forward(self, past):
        hidden = self.hidden_linear(past)
        hidden = self.activation(hidden)
        out = self.out_linear(hidden)
        out = self.activation(out)

        return out