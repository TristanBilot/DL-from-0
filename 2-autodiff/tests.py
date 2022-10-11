import torch
import numpy as np

from autodiff import Tensor


def _sigmoid_derivative(x: float):
    return x * (1 - x)

def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def test_1_layer_nn():

    W1 = Tensor(np.array([[-0.5,  0.5], [1.827, 0.213], [-2,  2], [2.34, 0.657]]))
    b1 = Tensor(np.array([[1.2],  [1.2], [1.2],  [1.2]]))
    X = Tensor(np.array([[-2], [0.122]]))

    sig = Tensor().sigmoid

    y = sig(W1 @ X + b1)
    y.backpropagate()

    # dy/dW
    assert np.all(np.round(W1.gradient) == np.round(np.array([
        [-0.17,  0.01],
        [-0.14,  0.00],
        [-0.00,  0.00],
        [-0.06,  0.00]
    ])))

    # dy/db
    assert np.all(np.round(b1.gradient) == np.round(np.array([
        [0.08549257],
        [0.07449035],
        [0.00428504],
        [0.03125697]
    ])))

    # Tests with chain rule
    W1_a, X_a, b1_a =  W1.val, X.val, b1.val
    wxb = _sigmoid(W1_a @ X_a + b1_a)
    
    # dy/dW1
    dw1_pred = np.dot(_sigmoid_derivative(wxb), X_a.T)
    assert np.all(np.round(W1.gradient) == np.round(dw1_pred))


    # Tests with torch
    W1_b = torch.tensor(W1.val.copy(), requires_grad=True)
    b1_b = torch.tensor(b1.val.copy(), requires_grad=True)
    X_b = torch.tensor(X.val.copy(), requires_grad=True)

    sig = torch.nn.Sigmoid()
    y = sig(W1_b @ X_b + b1_b)
    y.backward(torch.tensor([[1.],  [1.], [1.],  [1.]]))

    dw1 = W1_b.grad.cpu().detach().numpy()
    assert np.all(np.round(W1.gradient) == np.round(dw1))
