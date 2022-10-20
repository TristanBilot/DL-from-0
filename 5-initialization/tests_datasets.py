from sklearn.datasets import load_digits
import numpy as np

import autodiff as ad
import visualization as viz
from layers import Linear, MSE, Sigmoid, NllLoss, Argmax, CrossEntropyLoss
from models import MLP
from optimizers import Adam
from utils import generate_sin_dataset, shuffle_dataset, divide_train_test



def test_mnist_dataset_nll_loss():
    mnist = load_digits()
    X = np.array([image.flatten() for image in mnist.images]) / 255.
    y = np.array(mnist.target)
    X_train, X_test, y_train, y_test = divide_train_test(X, y, train_test=0.7, dtype_y=np.int32)

    softmax = ad.Tensor().softmax

    model = MLP(
        Linear(64, 100),
        Sigmoid(),
        Linear(100, 10),
        Sigmoid(),
        softmax,
    )

    optimizer = Adam(params=model.params, lr=0.001)
    loss_fn = NllLoss()

    accuracies, losses = [], []

    epochs = 800
    for i in range(epochs):
        X_train, X_test, y_train, y_test = shuffle_dataset(X_train, X_test, y_train, y_test)

        optimizer.zero_grad()

        loss, y_hat = model.train(X=X_train, y=y_train, optimizer=optimizer, loss_fn=loss_fn)
        
        sum = ad.Tensor().sum
        preds = Argmax()(y_hat, dim=1)
        correct_preds = sum(preds.eq(y_train)).val
        accuracy = correct_preds / y_train.shape[0]

        losses.append(loss.val)
        accuracies.append(accuracy)

        if i % 100 == 0:
            print(loss.val, accuracy)

    y_hat = np.asarray(model(X_test).val, np.float32)
    assert accuracy > 0.8


def test_mnist_dataset_mse():
    def one_hot(n, max):
        arr = [0] * max
        arr[n] = 1
        return arr

    mnist = load_digits()
    X = np.array([image.flatten() for image in mnist.images]) / 255.
    y = np.array([one_hot(n, 10) for n in mnist.target])
    X_train, X_test, y_train, y_test = divide_train_test(X, y, train_test=0.7)

    softmax = ad.Tensor().softmax

    model = MLP(
        Linear(64, 100),
        Sigmoid(),
        Linear(100, 10),
        Sigmoid(),
        softmax,
    )

    optimizer = Adam(params=model.params, lr=0.001)
    loss_fn = MSE()

    accuracies, losses = [], []
    epochs = 800
    for i in range(epochs):
        X_train, X_test, y_train, y_test = shuffle_dataset(X_train, X_test, y_train, y_test)

        optimizer.zero_grad()
        loss, y_hat = model.train(X=X_train, y=y_train, optimizer=optimizer, loss_fn=loss_fn)
        
        sum = ad.Tensor().sum
        argmax = Argmax()
        y_hat = argmax(y_hat, dim=1)
        truth = argmax(y_train, dim=1)
        correct_preds = sum(y_hat.eq(truth)).val
        accuracy = correct_preds / y_train.shape[0]

        losses.append(loss.val)
        accuracies.append(accuracy)

        if i % 100 == 0:
            print(loss.val, accuracy)

    y_hat = np.asarray(model(X_test).val, np.float32)
    assert accuracy > 0.8


def test_mnist_dataset_crossentropy():
    def one_hot(n, max):
        arr = [0] * max
        arr[n] = 1
        return arr

    mnist = load_digits()
    X = np.array([image.flatten() for image in mnist.images]) / 255.
    y = np.array([one_hot(n, 10) for n in mnist.target])
    X_train, X_test, y_train, y_test = divide_train_test(X, y, train_test=0.7)

    softmax = ad.Tensor().softmax

    model = MLP(
        Linear(64, 100),
        Sigmoid(),
        Linear(100, 10),
        Sigmoid(),
        softmax,
    )

    optimizer = Adam(params=model.params, lr=0.001)
    loss_fn = CrossEntropyLoss()

    accuracies, losses = [], []
    epochs = 800
    for i in range(epochs):
        X_train, X_test, y_train, y_test = shuffle_dataset(X_train, X_test, y_train, y_test)

        optimizer.zero_grad()
        loss, y_hat = model.train(X=X_train, y=y_train, optimizer=optimizer, loss_fn=loss_fn)
        
        sum = ad.Tensor().sum
        argmax = Argmax()
        y_hat = argmax(y_hat, dim=1)
        truth = argmax(y_train, dim=1)
        correct_preds = sum(y_hat.eq(truth)).val
        accuracy = correct_preds / y_train.shape[0]

        losses.append(loss.val)
        accuracies.append(accuracy)

        if i % 100 == 0:
            print(loss.val, accuracy)

    y_hat = np.asarray(model(X_test).val, np.float32)
    assert accuracy > 0.8



# def test_mnist_dataset_crossentropy2():
#     def one_hot(n, max):
#         arr = [0] * max
#         arr[n] = 1
#         return arr

#     mnist = load_digits()
#     X = np.array([image.flatten() for image in mnist.images]) / 255.
#     # y = np.array([one_hot(n, 10) for n in mnist.target])
#     y =  np.array(mnist.target)
#     X_train, X_test, y_train, y_test = divide_train_test(X, y, train_test=0.7, dtype_y=np.int32)

#     model = MLP(
#         Linear(64, 100),
#         Sigmoid(),
#         Linear(100, 1),
#         Sigmoid(),
#         ad.Tensor().softmax
#     )

#     optimizer = Adam(params=model.params, lr=0.001)
#     loss_fn = CrossEntropyLoss()

#     accuracies, losses = [], []
#     epochs = 800
#     for i in range(epochs):
#         X_train, X_test, y_train, y_test = shuffle_dataset(X_train, X_test, y_train, y_test)

#         optimizer.zero_grad()
#         loss, y_hat = model.train(X=X_train, y=y_train, optimizer=optimizer, loss_fn=loss_fn)
        
#         sum = ad.Tensor().sum
#         preds = Argmax()(y_hat, dim=1)
#         correct_preds = sum(preds.eq(y_train)).val
#         accuracy = correct_preds / y_train.shape[0]

#         losses.append(loss.val)
#         accuracies.append(accuracy)

#         if i % 100 == 0:
#             print(loss.val, accuracy)


test_mnist_dataset_crossentropy()