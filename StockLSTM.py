import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 32,
                 num_layers: int = 1,
                 num_classes: int = 3,
                 dropout: float = 0.3
                 ):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # nur letzter Zeitschritt

        out = self.layer_norm(out)
        out = self.dropout(out)
        out = self.fc(out)

        return out