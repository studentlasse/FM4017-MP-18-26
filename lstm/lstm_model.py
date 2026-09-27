import numpy as np
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """Initialize the LSTM model with the given parameters."""
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size # Number of features in the hidden state
        self.num_layers = num_layers # Number of lstm layers
        self.lst = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) # Define the lstm layer
        self.fc = nn.Linear(hidden_size, output_size) # Define the fully connected layer

    def forward(self, x, h0=None, c0=None):
        """Run the data through the model."""
        if h0 is None or c0 is None:
            # Initialize hidden state and cell state with zeros if not provided
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, (hn, cn) = self.lst(x, (h0, c0)) # Pass the input through the lstm layer
        out = self.fc(out[:, -1, :]) # Pass last time step
        return out, (hn, cn) 