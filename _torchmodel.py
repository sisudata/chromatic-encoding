# main.py
from time import time
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import tqdm

# uses my minor edits for value-based pytorchfm
# https://github.com/vlad17/pytorch-fm
from torchfm.model.afi import AutomaticFeatureInteractionModel
from torchfm.model.afm import AttentionalFactorizationMachineModel
from torchfm.model.dcn import DeepCrossNetworkModel
from torchfm.model.dfm import DeepFactorizationMachineModel
from torchfm.model.ffm import FieldAwareFactorizationMachineModel
from torchfm.model.fm import FactorizationMachineModel
from torchfm.model.fnfm import FieldAwareNeuralFactorizationMachineModel
from torchfm.model.fnn import FactorizationSupportedNeuralNetworkModel
from torchfm.model.lr import LogisticRegressionModel
from torchfm.model.ncf import NeuralCollaborativeFiltering
from torchfm.model.nfm import NeuralFactorizationMachineModel
from torchfm.model.pnn import ProductNeuralNetworkModel
from torchfm.model.wd import WideAndDeepModel
from torchfm.model.xdfm import ExtremeDeepFactorizationMachineModel
from torchfm.model.afn import AdaptiveFactorizationNetwork

from torchfm.dataset.sps import SparseDataset
from scipy import sparse as sps

from svmlight_loader_install import binprefix
import numpy as np
import os

def svm_dataset(path, quiet, pad=None):
    if not quiet:
        print('loading', path, 'assuming svm2bins has run')
    X, y = binprefix(path)
    if pad is not None:
        # because dimensions are sparse-encoded, test may have fewer columns than train
        assert X.shape[1] <= pad, (X.shape, 'pad', pad)
        assert X.getformat() == 'csr', X.getformat()
        X = sps.csr_matrix((X.data, X.indices, X.indptr),
                           shape=(X.shape[0], pad))
    field_dims_path = path + '.field_dims.txt'
    if os.path.exists(field_dims_path):
        field_dims = np.loadtxt(field_dims_path).astype(np.uint32)
        assert len(field_dims) >= X.shape[1], (field_dims.shape, X.shape)
        field_dims = field_dims[:X.shape[1]]
    else:
        field_dims = np.ones(X.shape[1], dtype=np.uint32)
    return SparseDataset(X, y, field_dims, quiet), X.shape

def get_model(name, dataset, embed_dim):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    field_dims = dataset.field_dims
    if name == 'lr':
        return LogisticRegressionModel(field_dims)
    elif name == 'fm':
        return FactorizationMachineModel(field_dims, embed_dim=embed_dim)
    elif name == 'ffm':
        return FieldAwareFactorizationMachineModel(field_dims, embed_dim=embed_dim)
    elif name == 'fnn':
        return FactorizationSupportedNeuralNetworkModel(field_dims, embed_dim=embed_dim, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'wd':
        return WideAndDeepModel(field_dims, embed_dim=embed_dim, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'ipnn':
        return ProductNeuralNetworkModel(field_dims, embed_dim=embed_dim, mlp_dims=(16,), method='inner', dropout=0.2)
    elif name == 'opnn':
        return ProductNeuralNetworkModel(field_dims, embed_dim=embed_dim, mlp_dims=(16,), method='outer', dropout=0.2)
    elif name == 'dcn':
        return DeepCrossNetworkModel(field_dims, embed_dim=embed_dim, num_layers=3, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'nfm':
        return NeuralFactorizationMachineModel(field_dims, embed_dim=embed_dim, mlp_dims=(64,), dropouts=(0.2, 0.2))
    elif name == 'ncf':
        # only supports MovieLens dataset because for other datasets user/item colums are indistinguishable
        assert isinstance(dataset, MovieLens20MDataset) or isinstance(dataset, MovieLens1MDataset)
        return NeuralCollaborativeFiltering(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2,
                                            user_field_idx=dataset.user_field_idx,
                                            item_field_idx=dataset.item_field_idx)
    elif name == 'fnfm':
        return FieldAwareNeuralFactorizationMachineModel(field_dims, embed_dim=embed_dim, mlp_dims=(64,), dropouts=(0.2, 0.2))
    elif name == 'dfm':
        return DeepFactorizationMachineModel(field_dims, embed_dim=embed_dim, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'xdfm':
        return ExtremeDeepFactorizationMachineModel(
            field_dims, embed_dim=embed_dim, cross_layer_sizes=(16, 16), split_half=False, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'afm':
        return AttentionalFactorizationMachineModel(field_dims, embed_dim=embed_dim, attn_size=16, dropouts=(0.2, 0.2))
    elif name == 'afi':
        return AutomaticFeatureInteractionModel(
             field_dims, embed_dim=embed_dim, atten_embed_dim=64, num_heads=2, num_layers=3, mlp_dims=(400, 400), dropouts=(0, 0, 0))
    elif name == 'afn':
        return AdaptiveFactorizationNetwork(
            field_dims, embed_dim=embed_dim, LNN_dim=1500, mlp_dims=(400,400,400), dropouts=(0, 0, 0))
    else:
        raise ValueError('unknown model name: ' + name)

def train(model, optimizer, data_loader, criterion, device, quiet=False):
    model.train()
    total_loss = 0
    nex = 0
    for i, (fields, values, target) in enumerate(tqdm.tqdm(data_loader, disable=(quiet or None))):
        fields, values, target = fields.to(device), values.to(device), target.to(device)
        y = model(fields, values)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        nex += 1
    return total_loss / nex


from sklearn.metrics import log_loss, accuracy_score

def test(model, data_loader, device, quiet=False):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, value, target in tqdm.tqdm(data_loader, disable=(quiet or None)):
            fields, value, target = fields.to(device), value.to(device), target.to(device)
            y = model(fields, value)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())

    return accuracy_score(targets, [x > 0.5 for x in predicts]), log_loss(targets, predicts)

def runall(train_path,
           test_path,
           device,
           model_name='fm',
           epoch=20,
           embed_dim=1,
           learning_rate=1e-3,
           batch_size=256, # was 2048 actually which i thought was a lot
           weight_decay=1e-6,
           quiet=True,
           tqdm_quiet=True):
    train_dataset, shape = svm_dataset(train_path, quiet)
    train_length = len(train_dataset)
    test_dataset, tshape = svm_dataset(test_path, quiet, pad=shape[1])
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
    model = get_model(model_name, train_dataset, embed_dim).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_epoch_losses = []
    t = time()
    for epoch_i in range(epoch):
        loss = train(model, optimizer, train_data_loader, criterion, device, tqdm_quiet)
        if not quiet:
            print('epoch:', 1 + epoch_i, 'of', epoch,
                  'train: logloss: {:7.4f}'.format(loss))

        train_epoch_losses.append(loss)

    t = time() - t

    emit = {}

    emit["train_epoch_logloss"] = train_epoch_losses

    emit["test_acc"], emit["test_logloss"] = test(model, test_data_loader, device, tqdm_quiet)
    if not quiet:
        print('test auc:', 'logloss: {:7.4f} acc {:7.4f}'
              .format(emit["test_logloss"], emit["test_acc"]))

    emit["train_acc"], emit["train_logloss"] = test(model, train_data_loader, device, tqdm_quiet)

    emit["train_examples"] = train_length
    emit["learner"] = model_name
    emit["budget"] = shape[1]
    emit["test_examples"] = len(test_dataset)
    emit["train_sec"] = t
    emit['num_params'] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return emit
