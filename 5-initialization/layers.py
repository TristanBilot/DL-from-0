import math
from typing import List
from autodiff import Parameter

from autodiff import Tensor
import init


class Layer:
    def __call__(self, *args, **kwargs) -> Tensor:
        pass


class WeightedLayer(Layer):
    params: List[Tensor]

    def init_params(self, nb_inp: int, nb_out: int):
        pass


class Linear(WeightedLayer):
    def __init__(self, nb_inp: int, nb_out: int) -> None:
        self.weight: Parameter = None
        self.bias: Parameter = None
        self.params: List[Parameter] = None

        self.init_params(nb_inp, nb_out)

    def __call__(self, X: Tensor):
        assert isinstance(X, Tensor), "Parameter `X` should be a Tensor"
        return X @ self.weight + self.bias

    def init_params(self, nb_inp: int, nb_out: int):
        self.weight = Parameter.zeros(nb_inp, nb_out)
        self.bias = Parameter.zeros(1, nb_out)
        self.params = [self.weight, self.bias]

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)


class MSE(Layer):
    def __call__(self, y_hat: Tensor, y: Tensor):
        loss_fn = Tensor().mse_loss
        return loss_fn(y_hat=y_hat, y=y)


class NllLoss(Layer):
    def __call__(self, y_hat: Tensor, y: Tensor, **kwargs):
        loss_fn = Tensor().nll_loss
        return loss_fn(y_hat=y_hat, y=y, **kwargs)

class CrossEntropyLoss(Layer):
    def __call__(self, y_hat: Tensor, y: Tensor):
        loss_fn = Tensor().crossentropy_loss
        return loss_fn(y_hat=y_hat, y=y)


class Argmax(Layer):
    def __call__(self, input: 'Tensor', dim: int=None):
        fn = Tensor().argmax
        return fn(input=input, dim=dim)


class Sigmoid(Layer):
    def __call__(self, X: Tensor):
        sig = Tensor().sigmoid
        return sig(X)


class ReLU(Layer):
    def __call__(self, X: Tensor):
        relu = Tensor().relu
        return relu(X)


class Softmax(Layer):
    def __call__(self, X: Tensor, **kwargs):
        softmax = Tensor().softmax
        return softmax(X, **kwargs)


class Sum(Layer):
    def __call__(self, X: Tensor, **kwargs):
        sum = Tensor().sum
        return sum(X, **kwargs)
