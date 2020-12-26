import torch
import numpy as np

from .embed import FeaturesEmbedding
from .layer import MultiLayerPerceptron


class WideAndDeepModel(torch.nn.Module):
    """
    A pytorch implementation of wide and deep learning.

    Reference:
        HT Cheng, et al. Wide & Deep Learning for Recommender Systems, 2016.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        # save an extra float for the "wide" portion
        self.linear = FeaturesEmbedding(field_dims, 1)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_dim = embed_dim
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(
            self.embed_output_dim, mlp_dims[:-1], mlp_dims[-1], dropout)
        assert sum(1 for _ in self.named_parameters()) == len(
            set(k for k, _ in self.named_parameters())), "not unique"

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``

        output float tensor logits of shape ``(batch_size,)``
        """
        # b = batch size
        # f = num fields
        # e = embed dim
        # m = mlp last layer dim
        embed_bfe = self.embedding(x)
        linear_bf = self.linear(x).squeeze(-1)
        linear_b = linear_bf.sum(-1)
        deep_b = self.mlp(embed_bfe.view(-1, self.embed_output_dim)).sum(-1)
        return linear_b + deep_b

    def sparse_parameters(self):
        for attr, val in self.named_parameters():
            if attr.startswith('linear.') or attr.startswith('embedding.'):
                yield attr, val

    def dense_parameters(self):
        sparse = set(k for k, _ in self.sparse_parameters())
        for attr, val in self.named_parameters():
            if attr not in sparse:
                yield attr, val
