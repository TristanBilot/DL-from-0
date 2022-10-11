from typing import List

from optimizers import Optimizer
from autodiff import Tensor


class Layer:
    def __call__(self, *args, **kwargs) -> Tensor:
        pass


class WeightedLayer(Layer):
    params: List[Tensor]

    def init_params(self, nb_inp: int, nb_out: int):
        pass


class Linear(WeightedLayer):
    def __init__(self, nb_inp: int, nb_out: int) -> None:
        self.weight = None
        self.bias = None
        self.params = None

        self.init_params(nb_inp, nb_out)

    def __call__(self, X: Tensor):
        return X @ self.weight + self.bias

    def init_params(self, nb_inp: int, nb_out: int):
        self.weight = Tensor.randn(nb_inp, nb_out)
        self.bias = Tensor.randn(1, nb_out)
        self.params = [self.weight, self.bias]


class MSE(Layer):
    def __call__(self, y_hat: Tensor, y: Tensor):
        loss_fn = Tensor().mse_loss
        return loss_fn(y_hat=y_hat, y=y)


class Sigmoid(Layer):
    def __call__(self, X: Tensor):
        sig = Tensor().sigmoid
        return sig(X)


class Model:
    def __init__(self, *layers) -> None:
        self.layers = layers

    def __call__(self, X: Tensor) -> Tensor:
        y_hat = X
        for layer in self.layers:
            y_hat = layer(y_hat)
        return y_hat

    def train(
        self,
        X: Tensor,
        y: Tensor,
        optimizer: Optimizer,
        loss_fn: Layer,
    ):
        y_hat = self(X)
        loss = loss_fn(y_hat, y)
        loss.backpropagate()

        optimizer.optimize()

    @property
    def params(self):
        params = []
        for l in self.layers:
            if isinstance(l, WeightedLayer):
                params += l.params
        return params

