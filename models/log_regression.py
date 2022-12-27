import torch
from torch import nn

class Log_reg(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Log_reg, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activ = nn.Sigmoid()

    def forward(self, x_0):
        x = self.linear(x_0)
        out = self.activ(x)
        return out


if __name__ == '__main__':
    log_reg = Log_reg(1280, 1)
    input = torch.rand(1280)
    out = log_reg(input)
    print(out)

    log_reg = Log_reg(1280, 1)
    input = torch.rand(5, 1280)
    out = log_reg(input)
    print(out)
