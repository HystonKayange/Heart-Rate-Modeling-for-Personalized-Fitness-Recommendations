import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim, dropout=0.5, bidirectional=True):
        super(LSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Bidirectional LSTM layer with specified input dimension, hidden dimension, number of layers, and dropout
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)
        
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer to project the LSTM output to the output dimension
        self.fc = nn.Linear(hidden_dim * self.num_directions, output_dim)
        
        # Optional: Learnable initial states for h0 and c0
        self.h0 = nn.Parameter(torch.zeros(n_layers * self.num_directions, hidden_dim))
        self.c0 = nn.Parameter(torch.zeros(n_layers * self.num_directions, hidden_dim))
        
        # Optional: Batch normalization before the final output
        self.batch_norm = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        # If using learnable initial states, expand to match batch size
        batch_size = x.size(0)
        h0 = self.h0.unsqueeze(1).repeat(1, batch_size, 1)
        c0 = self.c0.unsqueeze(1).repeat(1, batch_size, 1)
        
        # Passing the input through the LSTM layers
        out, _ = self.lstm(x, (h0, c0))
        
        # Applying dropout to the output of the LSTM's last time step
        out = self.dropout(out[:, -1, :])
        
        # Passing through the fully connected layer
        out = self.fc(out)
        
        # Optional: Batch normalization
        out = self.batch_norm(out)
        
        return out
