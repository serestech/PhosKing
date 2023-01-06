import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.init as init
import torch.nn.functional as F

class FFNN(nn.Module):
    def __init__(self, input_size, n_hidden_units, out_size):
        super(FFNN, self).__init__()
        
        self.input = nn.Linear(input_size, n_hidden_units)
        # self.input.weight = init.kaiming_normal_(self.input.weight)
        # self.input.bias = init.constant_(self.input.bias, 0)

        self.h1 = nn.Linear(n_hidden_units, n_hidden_units)
        # self.h1.weight = init.kaiming_normal_(self.h1.weight)
        # self.h1.bias = init.constant_(self.h1.bias, 0)
        
        self.h2 = nn.Linear(n_hidden_units, n_hidden_units)
        # self.h2.weight = init.kaiming_normal_(self.h2.weight)
        # self.h2.bias = init.constant_(self.h2.bias, 0)


        self.output = nn.Linear(n_hidden_units, out_size)
        # self.output.weight = init.kaiming_normal_(self.output.weight)
        # self.output.bias = init.constant_(self.output.bias, 0)

        self.activation = torch.nn.ReLU()
        self.out_activation = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=0.5, inplace=True)
        self.batchnorm = torch.nn.BatchNorm1d(n_hidden_units)

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.h1(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.h2(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.output(x)
        x = self.out_activation(x)
        return x

if __name__ == '__main__':
    ffnn = FFNN(1280, 2560, 1)
    input = torch.rand(1280)
    out = ffnn(input)
    print(out)
    
    ffnn = FFNN(1280, 2560, 2)
    input = torch.rand(5, 1280)
    out = ffnn(input)
    print(out)
