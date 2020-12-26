import numpy as np
import torch.utils.data

class SparseDataset(torch.utils.data.Dataset):
    """
    Accepts a CSR X with *integral* data.

    Each column will be interpreted as a feature with multiple fields
    (categorical).

    CSR is assumed to have no duplicate entries in a row and indices are
    to already be sorted. Field dims should be greater than X.
    """

    def __init__(self, X, y, field_dims):
        """
        We must have `np.all(field_dims <= X.max(axis=0))`.

        NN inputs will be len(field_dims) long.
        """
        self.y = y
        self.X = X
        assert self.X.getformat() == 'csr'
        assert self.X.dtype == np.uint32, self.X.dtype
        self.field_dims = np.asarray(field_dims, dtype=np.uint32)
        self.num_fields = len(self.field_dims)
        assert self.X.shape[1] == len(field_dims)

    # returns a two-tuple (field indices, target)
    # the first an array of length (num fields) of type np.long
    # the second a float
    # float values will usually be 1 for categorical data but may be different
    # for numeric data
    def __getitem__(self, index):

        start, stop = self.X.indptr[index], self.X.indptr[index + 1]
        col_idxs = self.X.indices[start:stop]
        nnz_vals = self.X.data[start:stop]

        fields = np.zeros(self.num_fields, dtype=np.long)
        fields[col_idxs] = nnz_vals

        return fields, float(self.y[index])

    def __len__(self):
        return len(self.y)
