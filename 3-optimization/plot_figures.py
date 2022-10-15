import math
import numpy as np
import torch

import autodiff as ad
import visualization as viz
from layers import Linear, MSE, Sigmoid, ReLU
from models import MLP
from optimizers import SGD, Adam


def plot_xor_layers_loss_acc():
    accuracies_, losses_ = [], []

    for i in range(6):
        X = ad.Tensor(np.array([
            [0., 0.],
            [0., 1.],
            [1., 0.],
            [1., 1.],
        ]))
        y = ad.Tensor(np.array([
            [1.],
            [0.],
            [1.],
            [0.],
        ]))
        model = MLP(
            Linear(2, 100),
            Sigmoid(),
            Linear(100, 1),
            Sigmoid(),
        )

        lr=0.01 - i *0.002
        # optimizer = SGD(params=model.params, lr=lr)
        optimizer = Adam(params=model.params, lr=0.01 - i *0.002)
        loss_fn = MSE()

        accuracies, losses = [], []

        epochs = 300
        for _ in range(epochs):
            loss, y_hat = model.train(X=X, y=y, optimizer=optimizer, loss_fn=loss_fn)
            
            # Measuring accuracy
            sum = ad.Tensor().sum
            correct_preds = sum(y_hat.near_eq(y)).val
            accuracy = correct_preds / y.shape[0]
            losses.append(loss.val)
            accuracies.append(accuracy)

        accuracies_.append((accuracies, f'lr={round(lr, 3)}'))
        losses_.append((losses, f'lr={round(lr, 3)}'))

        y_hat = model(X).val

    viz.plot_lists(losses_)
    viz.plot_lists(accuracies_)

plot_xor_layers_loss_acc()