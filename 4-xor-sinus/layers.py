from typing import List
from autodiff import Parameter

from autodiff import Tensor


class Layer:
    def __call__(self, *args, **kwargs) -> Tensor:
        pass


class WeightedLayer(Layer):
    params: List[Parameter]


class Linear(WeightedLayer):
    def __init__(self, nb_inp: int, nb_out: int) -> None:
        self.weight = Parameter.randn(nb_inp, nb_out)
        self.bias = Parameter.zeros(1, nb_out)
        self.params = [self.weight, self.bias]

    def __call__(self, X: Tensor):
        assert isinstance(X, Tensor), "Parameter `X` should be a Tensor"
        return X @ self.weight + self.bias

class MSE(Layer):
    def __call__(self, y_hat: Tensor, y: Tensor):
        loss_fn = Tensor().mse_loss
        return loss_fn(y_hat=y_hat, y=y)


class Sigmoid(Layer):
    def __call__(self, X: Tensor):
        sig = Tensor().sigmoid
        return sig(X)


class ReLU(Layer):
    def __call__(self, X: Tensor):
        relu = Tensor().relu
        return relu(X)
