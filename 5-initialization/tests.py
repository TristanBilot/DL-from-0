import math
import numpy as np
import torch

import autodiff as ad
import visualization as viz
from layers import Linear, MSE, Sigmoid, ReLU, NllLoss, Softmax, CrossEntropyLoss
from models import MLP
from optimizers import SGD, Adam
from utils import generate_sin_dataset, shuffle_dataset

np.random.seed(42)
np.set_printoptions(suppress=True)


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x: float):
    return sigmoid(x) * (1 - sigmoid(x))


def test_chain_rule_1_layer():
    """Compares the derivative of dy/dW1 from pytorch to
    the one from the chain rule.
    """
    W1 = np.array([[-0.5,  0.5], [1.827, 0.213], [-2,  2], [2.34, 0.657]])
    b1 = np.array([[1.2],  [1.2], [1.2],  [1.2]])
    X = np.array([[-2], [0.122]])

    wxb = W1 @ X + b1
    # dy/dW1
    dw1_pred = np.dot(sigmoid_derivative(wxb), X.T)

    W1 = torch.tensor(W1.copy(), requires_grad=True)
    b1 = torch.tensor(b1.copy(), requires_grad=True)
    X = torch.tensor(X.copy(), requires_grad=True)

    sig = torch.nn.Sigmoid()
    y = sig(W1 @ X + b1)
    y.backward(torch.tensor([[1.],  [1.], [1.],  [1.]]))

    # dy/dW2 in numpy array
    dw1 = W1.grad.cpu().detach().numpy()
    assert np.all(dw1_pred == dw1)


def test_chain_rule_2_layers():
    """Compares the derivative of dy/dW2 from pytorch to
    the one from the chain rule, with 2 hidden layers.
    """
    W1 = np.array([[-0.5,  0.5], [-2,  2]])
    b1 = np.array([[1.2,  1.2], [1.2,  1.2]])
    W2 = np.array([[-1.5,  1.5], [-2,  2]])
    b2 = np.array([[0.6,  0.7], [1.2,  1.2]])
    X = np.array([[-2,  2], [1.5, 1.5]])

    wxb = sigmoid(np.dot(W1, X) + b1)
    # dy/dW2
    dw2_pred = np.dot(sigmoid_derivative(np.dot(W2, wxb) + b2), wxb.T)

    W1 = torch.tensor(W1.copy(), requires_grad=True)
    b1 = torch.tensor(b1.copy(), requires_grad=True)
    W2 = torch.tensor(W2.copy(), requires_grad=True)
    b2 = torch.tensor(b2.copy(), requires_grad=True)
    X = torch.tensor(X.copy(), requires_grad=True)

    sig = torch.nn.Sigmoid()
    y = sig(W2 @ sig(W1 @ X + b1) + b2)
    y.backward(torch.tensor([[1., 1.], [1., 1.]]))

    # dy/dW2 in numpy array
    dw2 = W2.grad.cpu().detach().numpy()

    assert np.all(dw2_pred == dw2)


def test_sum_1():
    """Tests derivative of sum(x) w.r.t x
    """
    a = np.array([[-0.5], [-2]])
    # our
    y_hat = ad.Tensor(a)
    sum = ad.Tensor().sum
    res = sum(y_hat)
    res.backpropagate()

    # torch
    y_hat_torch = torch.tensor(a, requires_grad=True)
    mse = y_hat_torch.sum()
    mse.backward()

    dy_hat_w_torch = y_hat_torch.grad.cpu().detach().numpy()

    assert np.all(y_hat.gradient == dy_hat_w_torch)


def test_sum_2():
    """Tests mse loss as expression
    """
    a = np.array([[-0.5], [-2]])
    b = np.array([[1.2], [1.2]])
    # our
    y_hat = ad.Tensor(a)
    y = ad.Tensor(b)
    n = ad.Tensor(y_hat.val.size)
    sum = ad.Tensor().sum
    mse = (sum((y_hat - y) ** 2)) / n
    mse.backpropagate()

    # torch
    y_hat_torch = torch.tensor(a, requires_grad=True)
    y_torch = torch.tensor(b)
    n = len(y_hat_torch)
    mse = (((y_hat_torch - y_torch) ** 2).sum()) / n
    mse.backward()

    dy_hat_w_torch = np.asarray(y_hat_torch.grad.cpu().detach().numpy(), dtype=np.float32)

    assert np.all(y_hat.gradient == dy_hat_w_torch)


def test_mse_1():
    a = np.array([[-0.5], [-2]])
    b = np.array([[1.2], [1.2]])
    # our
    y_hat = ad.Tensor(a)
    y = ad.Tensor(b)
    mse = ad.Tensor().mse_loss

    c = mse(y_hat, y)
    c.backpropagate()

    # torch
    y_hat_torch = torch.tensor(a, requires_grad=True)
    y = torch.tensor(b, requires_grad=True)
    mse_func = torch.nn.MSELoss()

    mse = mse_func(y_hat_torch, y)
    mse.backward()

    dy_hat_w_torch =  np.asarray(y_hat_torch.grad.cpu().detach().numpy(), dtype=np.float32)

    assert np.all(y_hat.gradient == dy_hat_w_torch)


def test_1_layer_mse():
    """Tests a basic MLP with one hidden layer and MSE loss
    """
    a = np.array([[-0.5,  0.5], [-2,  2]])
    b = np.array([[1.2,  1.2], [1.2,  1.2]])
    c = np.array([[-2,  2], [1.5, 1.5]])
    d = np.array([[1., 1.], [1., 1.]])

    # our
    W1 = ad.Tensor(a)
    b1 = ad.Tensor(b)
    X = ad.Tensor(c)
    y = ad.Tensor(d)

    sig = ad.Tensor().sigmoid
    mse = ad.Tensor().mse_loss

    loss = mse(sig(W1 @ X + b1), y)
    loss.backpropagate()

    # torch
    W1_torch = torch.tensor(a, requires_grad=True)
    b1_torch = torch.tensor(b, requires_grad=True)
    X_torch = torch.tensor(c, requires_grad=True)
    y_torch = torch.tensor(d, requires_grad=True)

    sig_func = torch.nn.Sigmoid()
    mse_func = torch.nn.MSELoss()

    loss_torch = mse_func(sig_func(W1_torch @ X_torch + b1_torch), y_torch)
    loss_torch.backward()

    dy_hat_W1_torch = np.asarray(W1_torch.grad.cpu().detach().numpy(), dtype=np.float32)

    assert np.all(np.round(W1.gradient, 3) == np.round(dy_hat_W1_torch, 3))


def test_2_layers_mse():
    """Tests a basic MLP with two hidden layers and MSE loss
    """
    a = np.array([[-0.5,  0.5], [-2,  2]])
    b = np.array([[1.2,  1.2], [1.2,  1.2]])
    c = np.array([[-2,  2], [1.5, 1.5]])
    d = np.array([[1., 1.], [1., 1.]])

    e = np.array([[-1.5,  1.5], [-2,  2]])
    f = np.array([[0.6,  0.7], [1.2,  1.2]])

    # our
    W1 = ad.Tensor(a)
    b1 = ad.Tensor(b)
    W2 = ad.Tensor(e)
    b2 = ad.Tensor(f)
    X = ad.Tensor(c)
    y = ad.Tensor(d)

    sig = ad.Tensor().sigmoid
    mse = ad.Tensor().mse_loss

    layer1 = sig(W1 @ X + b1)
    loss = mse(sig(W2 @ layer1 + b2), y)
    loss.backpropagate()

    # torch
    W1_torch = torch.tensor(a, requires_grad=True)
    b1_torch = torch.tensor(b, requires_grad=True)
    W2_torch = torch.tensor(e, requires_grad=True)
    b2_torch = torch.tensor(f, requires_grad=True)
    X_torch = torch.tensor(c, requires_grad=True)
    y_torch = torch.tensor(d, requires_grad=True)

    sig_func = torch.nn.Sigmoid()
    mse_func = torch.nn.MSELoss()

    layer1_torch = sig_func(W1_torch @ X_torch + b1_torch)
    loss_torch = mse_func(sig_func(W2_torch @ layer1_torch + b2_torch), y_torch)
    loss_torch.backward()

    dy_hat_W2_torch = np.asarray(W2_torch.grad.cpu().detach().numpy(), dtype=np.float32)

    assert np.all(np.round(W2.gradient, 4) == np.round(dy_hat_W2_torch, 4))


def test_xor_with_tensors():
    """Tests a basic MLP with two hidden layers and MSE loss
    """
    # (4, 2)
    X = ad.Tensor(np.array([
        [0., 0.],
        [0., 1.],
        [1., 0.],
        [1., 1.],
    ]))
    # (4, 1)
    y = ad.Tensor(np.array([
        [1.],
        [0.],
        [1.],
        [0.],
    ]))
    
    W1 = ad.Tensor(np.random.randn(2, 50))
    W2 = ad.Tensor(np.random.randn(50, 1))
    b1 = ad.Tensor(np.random.randn(50))
    b2 = ad.Tensor(np.random.randn(1))

    sig = ad.Tensor().sigmoid
    mse = ad.Tensor().mse_loss

    epochs = 1000
    lr = 0.001
    for i in range(epochs + 1):
        # (4, 2) @ (2, 50) = (4, 50)
        layer1 = sig(X @ W1 + b1)
        # (4,50) @ (50, 1) = (4, 1)
        y_hat = sig(layer1 @ W2 + b2)

        loss = mse(y_hat, y)
        loss.backpropagate()

        W1.val = W1.val - lr * W1.gradient
        W2.val = W2.val - lr * W2.gradient
        b1.val = b1.val - lr * b1.gradient
        b2.val = b2.val - lr * b2.gradient

    def test(input):
        layer1 = sig(input @ W1 + b1)
        y_hat = sig(layer1 @ W2 + b2)
        return np.round(y_hat.val)

    assert np.all(test(X) == y.val)


def test_xor_with_layers():
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

    optimizer = SGD(params=model.params, lr=0.01)
    # optimizer = Adam(params=model.params)
    loss_fn = MSE()

    accuracies, losses = [], []

    epochs = 700
    for _ in range(epochs):
        loss, y_hat = model.train(X=X, y=y, optimizer=optimizer, loss_fn=loss_fn)
        
        # Measuring accuracy
        sum = ad.Tensor().sum
        correct_preds = sum(y_hat.near_eq(y, 0)).val
        accuracy = correct_preds / y.shape[0]
        losses.append(loss.val)
        accuracies.append(accuracy)

    y_hat = np.asarray(model(X).val, np .float32)
    assert np.all(np.round(y_hat) == y.val)


def test_sin():
    X_train, X_test, y_train, y_test = generate_sin_dataset(200)

    model = MLP(
        Linear(1, 200),
        ReLU(),
        Linear(200, 200),
        ReLU(),
        Linear(200, 1),
    )

    optimizer = Adam(params=model.params, lr=0.00005)
    loss_fn = MSE()

    epochs = 4000
    losses, accuracies = [], []

    def test(model):
        y_hat = model(X_test)
        sum = ad.Tensor().sum
        correct_preds = sum(y_hat.near_eq(y_test)).val
        accuracy = correct_preds / y_test.shape[0]
        return accuracy

    for i in range(epochs):
        X_train, X_test, y_train, y_test = shuffle_dataset(X_train, X_test, y_train, y_test)

        optimizer.zero_grad()
        loss, y_hat = model.train(X=X_train, y=y_train, optimizer=optimizer, loss_fn=loss_fn)
        losses.append(loss.val)

        sum = ad.Tensor().sum
        correct_preds = sum(y_hat.near_eq(y_train)).val
        accuracy = correct_preds / y_train.shape[0]
        accuracies.append(accuracy)

        if i % 200 == 0:
            test_acc = test(model)
            print(test_acc)

    assert accuracy > 0.5


def test_mlp_mutiple_input_sizes():
    model = MLP(
        Linear(10, 50),
        Linear(50, 1),
    )

    A = ad.Tensor(np.random.randn(32, 10))
    res = model(A)
    res = MSE()(res, ad.Tensor(np.zeros_like(res)))
    res.backpropagate()

    B = ad.Tensor(np.random.randn(60, 10))
    res = model(B)
    res = MSE()(res, ad.Tensor(np.zeros_like(res)))
    res.backpropagate()


def mock_torch_dataset(inp: int, out: int, batch_size: int):
    X = np.random.randn(batch_size, inp)
    y = np.random.randint(0, out, (batch_size,))
    return ad.Tensor(X), ad.Tensor(y, dtype=np.int32), torch.tensor(X), torch.tensor(y)


def train_torch(
    model,
    X: torch.Tensor,
    y: torch.Tensor,
    optimizer,
    loss_fn,
):
    y_hat = model(X.float())
    loss = loss_fn(y_hat, y)
    loss.backward()
    optimizer.step()
    return loss, y_hat


def test_nll():
    np.random.seed(42)
    inp, out, batch_size = 64, 10, 32
    X, y, X_t, y_t = mock_torch_dataset(inp, out, batch_size)
    lr = 0.001

    # Our
    model = MLP(
        Linear(inp, 100),
        Sigmoid(),
        Linear(100, out),
        Softmax()
    )
    optimizer = Adam(params=model.params, lr=lr)
    loss_fn = NllLoss()
    loss, y_hat = model.train(X=X, y=y, optimizer=optimizer, loss_fn=loss_fn)

    X2, y2, _, _ = mock_torch_dataset(inp, out, 99)
    model(X2)


def test_depth_1():
    np.random.seed(42)
    inp, out, batch_size = 1, 6, 32
    X, y, X_t, y_t = mock_torch_dataset(inp, out, batch_size)
    lr = 0.001

    model = MLP(
        Linear(1, 2),
        Sigmoid(),
        Linear(2, 3),
        Linear(3, 4),
        Linear(4, 5),
        Linear(5, 6),
        ReLU(),
        Sigmoid(),
        Softmax()
    )
    optimizer = Adam(params=model.params, lr=lr)
    loss_fn = NllLoss()
    _, _ = model.train(X=X, y=y, optimizer=optimizer, loss_fn=loss_fn)

    X2, y2, _, _ = mock_torch_dataset(inp, out, 99)
    model(X2)
    

test_mlp_mutiple_input_sizes()