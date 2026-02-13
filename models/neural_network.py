import torch
import torch.nn as nn
import numpy as np

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

class CNN1D(nn.Module):
    """
    Implements a 1D Convolutional Neural Network with configurable convolutional layers,
    ReLU activations, max pooling, dropout regularization, and sigmoid output for binary classification.
    
    Designed for sequential data such as time series or ECG signals.

    Attributes
    ----------
    conv_layers : nn.Sequential
        Sequential container of convolutional layers, ReLU activations, 
        max pooling, and dropout layers.
    fc_layers : nn.Sequential
        Sequential container of fully connected layers for classification.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Performs a forward pass through the network.
    """

    def __init__(
        self, 
        input_dim: int, 
        conv_channels: list = [32, 64, 128], 
        kernel_sizes: list = [7, 5, 3],
        fc_dims: list = [64, 32],
        dropout: float = 0.2
    ):
        """
        Initializes the CNN1D model.

        Parameters
        ----------
        input_dim : int
            Length of input sequence (number of time steps).
        conv_channels : list of int, optional
            List of output channels for each convolutional layer (default is [32, 64, 128]).
        kernel_sizes : list of int, optional
            List of kernel sizes for each convolutional layer (default is [7, 5, 3]).
        fc_dims : list of int, optional
            List of fully connected layer dimensions (default is [64, 32]).
        dropout : float, optional
            Dropout probability for regularization (default is 0.2).
        """
        super(CNN1D, self).__init__()
        
        # Build convolutional layers
        conv_layers = []
        in_channels = 1  # Single channel for univariate time series
        
        for out_channels, kernel_size in zip(conv_channels, kernel_sizes):
            conv_layers.append(nn.Conv1d(
                in_channels, 
                out_channels, 
                kernel_size=kernel_size, 
                padding=kernel_size//2
            ))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool1d(kernel_size=2))
            conv_layers.append(nn.Dropout(dropout))
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Calculate the size after convolutions and pooling
        # Each MaxPool1d with kernel_size=2 halves the sequence length
        conv_output_length = input_dim // (2 ** len(conv_channels))
        flattened_size = conv_channels[-1] * conv_output_length
        
        # Build fully connected layers
        fc_layers = []
        prev_dim = flattened_size
        
        for fc_dim in fc_dims:
            fc_layers.append(nn.Linear(prev_dim, fc_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout))
            prev_dim = fc_dim
        
        # Output layer
        fc_layers.append(nn.Linear(prev_dim, 1))
        fc_layers.append(nn.Sigmoid())
        
        self.fc_layers = nn.Sequential(*fc_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim) or (batch_size, 1, input_dim).

        Returns
        -------
        torch.Tensor
            The model's predictions as a squeezed tensor of shape (batch_size,)
            with values in [0, 1] after sigmoid activation.
        """
        # Add channel dimension if not present
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, input_dim) -> (batch_size, 1, input_dim)
        
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x.squeeze()