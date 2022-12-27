import torch
from torch import nn

class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(FFNN, self).__init__()
        self.linear_1 = nn.Linear(input_dim, hidden_dim)
        self.activ_1 = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_dim, output_dim)
        self.activ_2 = nn.Sigmoid()

    def forward(self, x_0):
        x = self.linear_1(x_0)
        x = self.activ_1(x)
        x = self.linear_2(x)
        out = self.activ_2(x)
        return out
    
if __name__ == '__main__':
    ffnn = FFNN(1280, 2000, 1)
    input = torch.rand(1280)
    out = ffnn(input)
    print(out)

    log_reg = FFNN(1280, 1)
    input = torch.rand(5, 1280)
    out = log_reg(input)
    print(out)
