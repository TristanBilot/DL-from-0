import numpy as np
import torch


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

