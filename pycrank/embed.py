"""
Embedding layers.
"""

import numpy as np
import torch

class FeaturesEmbedding(torch.nn.Module):
    """
    Expects "nullable fields", i.e., for the 1D positive int array
    `field_dims` this was
    initialized with, expects long input of length `num_fields`
    where `num_fields = len(field_dims)` for each batch item.

    Each value at index `i` should be between `[0, field_dims[i]]`,
    inclusive, where `0` means null (the all-zero embedding).

    This means non-null values are 1-indexed and the non-null cardinality
    of each field is equal to the maximum field value.
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embedding = torch.nn.Embedding(
            1 + sum(field_dims), embed_dim, padding_idx=0, sparse=True)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``

        note zeros stay zeros in every field
        returns a float tensor of shape `(batch_size, num_fields, embed_dim)`
        """
        # b = batch size
        # f = num fields
        # e = embed dim
        zero_bf = x == 0
        ixs_bf = x + x.new_tensor(self.offsets)
        ixs_bf[zero_bf] = 0
        return self.embedding(ixs_bf)
