import torch
from torch import nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes: int = 1):
        super(CNN, self).__init__()
        
        self.activ_fn = nn.ReLU()
        self.out_activation = nn.Sigmoid()

        self.conv1 = nn.Conv2d(1, 12, 2)
        self.conv2 = nn.Conv2d(12, 8, 3)

        self.fc1 = nn.Linear(40864, 15000)
        self.fc2 = nn.Linear(15000, 7500)
        self.fc3 = nn.Linear(7500, num_classes)
        

    def forward(self, x):
        x = self.activ_fn(self.conv1(x))
        x = self.activ_fn(self.conv2(x))

        x = torch.flatten(x, start_dim=1)

        x = self.activ_fn(self.fc1(x))
        x = self.activ_fn(self.fc2(x))
        x = self.fc3(x)
        
        x = self.out_activation(x)

        return x

if __name__ == '__main__':
    cnn = CNN()
    input = torch.rand(1, 1, 7, 1280)
    print('Running CNN...')
    out = cnn(input)
    print(f'{out=}')
    
    input = torch.rand(3, 1, 7, 1280)
    print('Running CNN...')
    out = cnn(input)
    print(f'{out=}')
    