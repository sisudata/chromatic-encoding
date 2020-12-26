"""
Common layers for NN construction.
"""

import torch

class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, hidden_dims, out_dim, dropout):
        super().__init__()
        layers = list()
        for embed_dim in hidden_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        layers.append(torch.nn.Linear(input_dim, out_dim))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)
