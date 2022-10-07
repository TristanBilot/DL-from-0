import numpy as np
import torch
import autodiff as ad


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

    dy_hat_w_torch = y_hat_torch.grad.cpu().detach().numpy()

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

    dy_hat_w_torch = y_hat_torch.grad.cpu().detach().numpy()

    assert np.all(y_hat.gradient == dy_hat_w_torch)
