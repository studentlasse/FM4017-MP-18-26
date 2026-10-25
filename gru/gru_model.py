import numpy as np
import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """Initialize the GRU model with the given parameters."""
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size # Number of features in the hidden state
        self.num_layers = num_layers # Number of lstm layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True) 
        self.fc = nn.Linear(hidden_size, output_size) 

    def forward(self, x, h0=None):
        """Run the data through the model."""
        if h0 is None:
            # Initialize hidden state with zeros
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, hn = self.gru(x, h0)
        out = self.fc(out[:, -1, :]) # Pass last time step
        return out, hn