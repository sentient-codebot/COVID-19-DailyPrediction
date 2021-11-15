import torch 
import torch.nn as nn

class ARModel(nn.Module):
    def __init__(self, p=1):
        super().__init__()
        self.bias = nn.parameter.Parameter(data=torch.zeros((1, 1)))
        self.a = nn.parameter.Parameter(data=torch.randn((1, p)))

    def forward(self, x_past):
        '''
        x(n) = - <x_past, self.a> + self.bias
        x_past (BATCH, p)
        self.a (1, p)
        self.bias (1, 1)

        return x_pred (BATCH, 1)
        '''
        return - torch.sum(x_past*self.a, dim=1).reshape(x_past.shape[0], 1) + self.bias


def main():
    p = 2
    ar_model = ARModel(p)
    x_past = torch.rand((10,p))
    x_n = torch.rand((10,1))

    x_pred = ar_model(x_past)


if __name__ == "__main__":
    main()