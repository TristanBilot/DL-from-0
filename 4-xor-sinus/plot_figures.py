import matplotlib.pyplot as plt
import numpy as np

import autodiff as ad
import visualization as viz
from layers import Linear, MSE, Sigmoid, ReLU
from models import MLP
from optimizers import SGD, Adam
from utils import generate_sin_dataset, shuffle_dataset


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


def plot_sin_layers_loss_acc():
    accuracies_, losses_, test_accuracies_ = [], [], []

    for i in range(6):
        X_train, X_test, y_train, y_test = generate_sin_dataset(200, train_test=0.5)

        model = MLP(
            Linear(1, 500),
            ReLU(),
            Linear(500, 1),
        )

        lr=0.0008 - i * 0.00002
        optimizer = Adam(params=model.params, lr=lr)
        loss_fn = MSE()

        epochs = 10000
        losses, accuracies, test_accuracies = [], [], []

        def test(model):
            y_hat = model(X_test)
            sum = ad.Tensor().sum
            correct_preds = sum(y_hat.near_eq(y_test, round=1)).val
            accuracy = correct_preds / y_test.shape[0]
            
            return accuracy

        def plot(model):
            y_hat = model(X_test)

            plt.scatter(X_test.val, y_hat.val, color='red', label="Predictions")
            plt.scatter(X_test.val, y_test.val, color='green', label="Truth")            
            
            plt.legend(loc="upper right")
            plt.show()

        for i in range(epochs):
            X_train, X_test, y_train, y_test = shuffle_dataset(X_train, X_test, y_train, y_test)

            optimizer.zero_grad()

            loss, y_hat = model.train(X=X_train, y=y_train, optimizer=optimizer, loss_fn=loss_fn)
            losses.append(loss.val)

            sum = ad.Tensor().sum
            correct_preds = sum(y_hat.near_eq(y_train, round=1)).val
            accuracy = correct_preds / y_train.shape[0]
            
            test_acc = test(model)
            accuracies.append(accuracy)
            test_accuracies.append(test_acc)

            if i % 200 == 0:
                print(test_acc, accuracy)

        # plot(model)
        accuracies_.append((accuracies, f'lr={round(lr, 5)}'))
        losses_.append((losses, f'lr={round(lr, 5)}'))
        test_accuracies_.append((test_accuracies, f'lr={round(lr, 5)}'))

    viz.plot_lists(losses_)
    viz.plot_lists(accuracies_)
    viz.plot_lists(test_accuracies_)

plot_sin_layers_loss_acc()