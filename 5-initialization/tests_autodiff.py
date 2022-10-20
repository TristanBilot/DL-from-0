import numpy as np
import torch

import autodiff as ad
from layers import Sum, Sigmoid, ReLU, NllLoss, CrossEntropyLoss, MSE, Softmax
from optimizers import SGD


def mock_tensors(val, dtype1=np.float32, dtype2=torch.float32, requires_grad=True):
    return ad.Tensor(val, dtype=dtype1), \
        torch.tensor(val, requires_grad=requires_grad, dtype=dtype2)

def equals(t1: ad.Tensor, t2: torch.Tensor, r: int=3):
    t1 = t1 if isinstance(t1, ad.Tensor) else ad.Tensor(t1)
    val_ok = \
        np.array_equal(np.round(np.asarray(t1.val, dtype=np.float32), r), \
        np.round(t2.cpu().detach().numpy().reshape(t1.shape), r))
    grad_ok = np.array_equal(np.round(t1.gradient, r), np.round(t2.grad.cpu().detach().numpy(), r)) \
        if t2.grad is not None else np.array_equal(np.zeros_like(t1.gradient, dtype=np.float32), t1.gradient)
    return val_ok and grad_ok


def test_tensors_add():
    a, a_t = mock_tensors(5)

    y = a * +1
    y_t = a_t * +1
    assert equals(y, y_t)

    y = +1 * y
    y_t = +1 * y_t
    assert equals(y, y_t)

    y = y + 1
    y_t = y_t + 1
    assert equals(y, y_t)

    y = +1 + y
    y_t = +1 + y_t
    assert equals(y, y_t)

    b, b_t = mock_tensors(-10)

    y = (b + y + b) + (b + y * 3)
    y_t = (b_t + y_t + b_t) + (b_t + y_t * 3)
    assert equals(y, y_t)

def test_tensors_sub():
    a, a_t = mock_tensors(5)

    y = -a
    y_t = -a_t
    assert equals(y, y_t)

    y = y - 1
    y_t = y_t - 1
    assert equals(y, y_t)

    y = -1 - y
    y_t = -1 - y_t
    assert equals(y, y_t)

    b, b_t = mock_tensors(-10)

    y = (b - y - b) - (b - y * 3)
    y_t = (b_t - y_t - b_t) - (b_t - y_t * 3)
    assert equals(y, y_t)


def test_tensors_mul():
    a, a_t = mock_tensors(5)

    y = a * -1
    y_t = a_t * -1
    assert equals(y, y_t)

    y = -1 * y
    y_t = -1 * y_t
    assert equals(y, y_t)

    y = y * y * y
    y_t = y_t * y_t * y_t
    assert equals(y, y_t)

    b, b_t = mock_tensors(-10)

    y = (b * y * b) * (b * y * 3)
    y_t = (b_t * y_t * b_t) * (b_t * y_t * 3)
    assert equals(y, y_t)


def test_tensors_div():
    a, a_t = mock_tensors(5)

    y = a / -1
    y_t = a_t / -1
    assert equals(y, y_t)

    y = -1 / y
    y_t = -1 / y_t
    assert equals(y, y_t)

    y = y / y / y
    y_t = y_t / y_t / y_t
    assert equals(y, y_t)

    y = -1 / y / -1 / y
    y_t = -1 / y_t / -1 / y_t
    assert equals(y, y_t)

    y = (2 / y / 2) / y
    y_t = (2 / y_t / 2) / y_t
    assert equals(y, y_t)

    y = (2 / y / 2) / (2 / y * 3)
    y_t = (2 / y_t / 2) / (2 / y_t * 3)
    assert equals(y, y_t)

    b, b_t = mock_tensors(-10)

    y = (b / y / b) / (b / y * 3)
    y_t = (b_t / y_t / b_t) / (b_t / y_t * 3)
    assert equals(y, y_t)


def test_tensors_pow():
    a, a_t = mock_tensors(0.3)

    y = a ** -1
    y_t = a_t ** -1
    assert equals(y, y_t)

    y = -1 ** y
    y_t = -1 ** y_t
    assert equals(y, y_t)

    y = y ** y ** y
    y_t = y_t ** y_t ** y_t
    assert equals(y, y_t)

    y = -1 ** y ** -1 ** y
    y_t = -1 ** y_t ** -1 ** y_t
    assert equals(y, y_t)

    y = (2 ** y ** 2) ** y
    y_t = (2 ** y_t ** 2) ** y_t
    assert equals(y, y_t)

    y = (2 ** y ** 2) ** (2 ** y * 3)
    y_t = (2 ** y_t ** 2) ** (2 ** y_t * 3)
    assert equals(y, y_t)


def test_tensors_pow():
    a, a_t = mock_tensors(0.3)

    y = a ** -1
    y_t = a_t ** -1
    assert equals(y, y_t)

    y = -1 ** y
    y_t = -1 ** y_t
    assert equals(y, y_t)

    y = y ** y ** y
    y_t = y_t ** y_t ** y_t
    assert equals(y, y_t)

    y = -1 ** y ** -1 ** y
    y_t = -1 ** y_t ** -1 ** y_t
    assert equals(y, y_t)

    y = (2 ** y ** 2) ** y
    y_t = (2 ** y_t ** 2) ** y_t
    assert equals(y, y_t)

    y = (2 ** y ** 2) ** (2 ** y * 3)
    y_t = (2 ** y_t ** 2) ** (2 ** y_t * 3)
    assert equals(y, y_t)


def test_tensors_matmul():
    a, a_t = mock_tensors(np.random.randn(10, 30), dtype1=np.float64, dtype2=torch.float64)
    b, b_t = mock_tensors(np.random.randn(30, 10), dtype1=np.float64, dtype2=torch.float64)

    y = a @ b @ a @ b
    y_t = a_t @ b_t @ a_t @ b_t

    assert equals(y, y_t, r=0)
    assert a.val.shape == a.gradient.shape
    assert b.val.shape == b.gradient.shape


def test_tensors_neg():
    a, a_t = mock_tensors(5)

    y = -a
    y_t = -a_t
    assert equals(y, y_t)

    y = -a - a - a
    y_t = -a_t - a_t - a_t
    assert equals(y, y_t)


def test_tensors_sum():
    a, a_t = mock_tensors(np.random.randn(10))
    sum = Sum()

    y = sum(a)
    y_t = a_t.sum()
    assert equals(y, y_t)

    a, a_t = mock_tensors(np.random.randn(10, 5))

    y = sum(a)
    y_t = a_t.sum()
    assert equals(y, y_t)

    a, a_t = mock_tensors(np.random.randn(10, 5))

    y = sum(a, axis=1)
    y_t = a_t.sum(dim=1)
    assert equals(y, y_t)

    a, a_t = mock_tensors(np.random.randn(10, 5, 4, 3))

    y = sum(a, axis=1)
    y_t = a_t.sum(dim=1)
    assert equals(y, y_t)

    a, a_t = mock_tensors(np.random.randn(10, 5, 4, 3))

    y = sum(a, axis=1, keepdims=True)
    y_t = a_t.sum(dim=1, keepdim=True)
    assert equals(y, y_t)


def test_tensors_sigmoid():
    a, a_t = mock_tensors(np.random.randn(10))
    sig = Sigmoid()
    sig_t = torch.nn.Sigmoid()

    y = sig(a)
    y_t = sig_t(a_t)
    assert equals(y, y_t)

    a, a_t = mock_tensors(np.random.randn(10, 5))

    y = sig(a)
    y_t = sig_t(a_t)
    assert equals(y, y_t)

    a, a_t = mock_tensors(np.random.randn(10, 5))

    y = sig(a)
    y_t = sig_t(a_t)
    assert equals(y, y_t)

    a, a_t = mock_tensors(np.random.randn(10, 5, 4, 3))

    y = sig(a)
    y_t = sig_t(a_t)
    assert equals(y, y_t)

    a, a_t = mock_tensors(np.random.randn(10, 5, 4, 3))

    y = sig(a)
    y_t = sig_t(a_t)
    assert equals(y, y_t)


def test_tensors_relu():
    a, a_t = mock_tensors(np.random.randn(10))
    relu = ReLU()
    relu_t = torch.nn.ReLU()

    y = relu(a)
    y_t = relu_t(a_t)
    assert equals(y, y_t)

    a, a_t = mock_tensors(np.random.randn(10, 5))

    y = relu(a)
    y_t = relu_t(a_t)
    assert equals(y, y_t)

    a, a_t = mock_tensors(np.random.randn(10, 5))

    y = relu(a)
    y_t = relu_t(a_t)
    assert equals(y, y_t)

    a, a_t = mock_tensors(np.random.randn(10, 5, 4, 3))

    y = relu(a)
    y_t = relu_t(a_t)
    assert equals(y, y_t)

    a, a_t = mock_tensors(np.random.randn(10, 5, 4, 3))

    y = relu(a)
    y_t = relu_t(a_t)
    assert equals(y, y_t)


def test_nll_loss():
    batch, inp, out = 1, 10, 1
    W1, W1_t = mock_tensors(np.random.randn(inp, 20))
    b1, b1_t = mock_tensors(np.random.randn(1, 20))
    W2, W2_t = mock_tensors(np.random.randn(20, out))
    b2, b2_t = mock_tensors(np.random.randn(1, out))

    X, X_t = mock_tensors(np.random.randn(batch, inp))
    y, y_t = mock_tensors(np.random.randint(0, out, (batch,)), \
        dtype1=np.int32, dtype2=torch.int32, requires_grad=False)

    # Our
    sig = Sigmoid()
    softmax = Softmax()
    loss_fn = NllLoss()
    y1 = sig(X @ W1 + b1)
    y_hat = softmax(sig(y1 @ W2 + b2))
    loss = loss_fn(y_hat, y, reduction='mean')
    loss.backpropagate()

    # Torch
    sig = torch.nn.Sigmoid()
    softmax = torch.nn.Softmax()
    loss_fn = torch.nn.NLLLoss(reduce=True, reduction='mean')
    y1_t = sig(X_t @ W1_t + b1_t)
    y_hat_t = softmax(sig(y1_t @ W2_t + b2_t))
    loss_t = loss_fn(y_hat_t, y_t.long())
    loss_t.backward()

    assert equals(y_hat, y_hat_t)
    assert equals(loss, loss_t)
    assert equals(W1.gradient, W1_t.grad)
    assert equals(W2.gradient, W2_t.grad)
    assert equals(b1.gradient, b1_t.grad)
    assert equals(b2.gradient, b2_t.grad)



# def test_crossentropy_loss():
#     batch, inp, out = 32, 10, 10
#     W1, W1_t = mock_tensors(np.random.randn(inp, 20))
#     b1, b1_t = mock_tensors(np.random.randn(1, 20))
#     W2, W2_t = mock_tensors(np.random.randn(20, out))
#     b2, b2_t = mock_tensors(np.random.randn(1, out))

#     X, X_t = mock_tensors(np.random.randn(batch, inp))
#     y, y_t = mock_tensors(np.random.randint(0, out, (batch,)), \
#         dtype1=np.int32, dtype2=torch.int32, requires_grad=False)

#     # Our
#     sig = Sigmoid()
#     # softmax = Softmax()
#     loss_fn = CrossEntropyLoss()
#     y1 = sig(X @ W1 + b1)
#     y_hat = sig(y1 @ W2 + b2)
#     loss = loss_fn(y_hat, y)
#     loss.backpropagate()

#     # Torch
#     sig = torch.nn.Sigmoid()
#     # softmax = torch.nn.
#     loss_fn = torch.nn.CrossEntropyLoss()
#     y1_t = sig(X_t @ W1_t + b1_t)
#     y_hat_t = sig(y1_t @ W2_t + b2_t)
#     loss_t = loss_fn(y_hat_t, y_t.long())
#     loss_t.backward()

#     assert equals(y_hat, y_hat_t)
#     assert equals(loss, loss_t)
#     assert equals(W1.gradient, W1_t.grad)
#     assert equals(W2.gradient, W2_t.grad)
#     assert equals(b1.gradient, b1_t.grad)
#     assert equals(b2.gradient, b2_t.grad)



# test_crossentropy_loss()