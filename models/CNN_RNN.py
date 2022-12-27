import torch
import torch.nn as nn

class CNN_RNN_FFNN(nn.Module):
    def __init__(self, n_input, hidden_linear_1, hidden_linear_2, dropout=0.5):
        super().__init__()
        kernel, padding = 7, 3
        out_channels = 48
        self.cnn_1 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv1d(in_channels=n_input, out_channels=out_channels,
            kernel_size=kernel, padding=padding),
            nn.ReLU()
        )

        kernel, padding = 13, 6
        self.cnn_2 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv1d(in_channels=n_input, out_channels=out_channels,
            kernel_size=kernel, padding=padding),
            nn.ReLU()
        )

        self.batchnorm = nn.BatchNorm1d(n_input + 2*out_channels)

        self.rnn = nn.LSTM(input_size=n_input + 2*out_channels, hidden_size=hidden_linear_1,
                           batch_first=True, num_layers=2, bidirectional=True, dropout=dropout)

        self.lnn = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_linear_1*2, hidden_linear_2),
            nn.ReLU(),
            nn.Linear(hidden_linear_2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x_0):
        x_0 = x_0.permute(0, 2, 1)
        x_1 = self.cnn_1(x_0)
        x_2 = self.cnn_2(x_0)
        x = torch.cat((x_0, x_1, x_2), dim=1)

        x = self.batchnorm(x)
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)
        
        x = x[:, x.shape[1]//2]
        out = self.lnn(x)
        return out
        

if __name__ == '__main__':
    cnn_rnn_ffnn = CNN_RNN_FFNN(1280, 512, 1024)
    input = torch.rand(5, 15, 1280)
    print('Running CNN_RNN_FFNN...')
    out = cnn_rnn_ffnn(input)
    print(f'{out=}')