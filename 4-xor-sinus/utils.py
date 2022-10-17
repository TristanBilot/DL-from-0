import numpy as np
import autodiff as ad


def generate_sin_data(n: int):
    dataset = np.empty([n, 2])
    data = 2 * np.pi * np.random.random_sample((n))
    dataset[:,0] = data
    dataset[:,1] = np.sin(data)
    X = dataset[:,0]
    y =  dataset[:,1]
    
    return X, y


def generate_sin_dataset(n: int, train_test: float=0.7):
    X, y = generate_sin_data(n)
    X_train, X_test = \
        ad.Tensor(X[:int(len(X) * train_test)].reshape(-1, 1)), \
        ad.Tensor(X[int(len(X) * train_test):].reshape(-1, 1))
    y_train, y_test = \
        ad.Tensor(y[:int(len(y) * train_test)].reshape(-1, 1)), \
        ad.Tensor(y[int(len(y) * train_test):].reshape(-1, 1))
    
    return X_train, X_test, y_train, y_test


def shuffle_dataset(
    X_train: ad.Tensor,
    X_test: ad.Tensor,
    y_train: ad.Tensor,
    y_test: ad.Tensor,
):
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train = X_train.val[indices]
    y_train = y_train.val[indices]

    indices = np.arange(X_test.shape[0])
    np.random.shuffle(indices)
    X_test = X_test.val[indices]
    y_test = y_test.val[indices]

    X_train, X_test = ad.Tensor(X_train), ad.Tensor(X_test)
    y_train, y_test = ad.Tensor(y_train), ad.Tensor(y_test)

    return X_train, X_test, y_train, y_test